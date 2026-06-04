"""
agents/chess_agent.py
A chess-playing agent with board-analysis tools, a persistent scratchpad,
and robust move extraction.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Callable

import chess

# --------------------------------------------------------------------------- #
# Piece labels for grouped move display                                        #
# --------------------------------------------------------------------------- #
_PIECE_LABEL = {
    chess.PAWN:   "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK:   "Rook",
    chess.QUEEN:  "Queen",
    chess.KING:   "King",
}

def _group_legal_moves(board: chess.Board, legal_moves: list[str]) -> str:
    """
    Return legal moves grouped by piece type so the LLM can easily find
    all options for a piece it wants to move.

    Example output:
        Pawn  : e2e4, e2e3, d2d4, d2d3
        Knight: g1f3, g1h3, b1c3, b1a3
        King  : (none)
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for uci in legal_moves:
        try:
            move  = chess.Move.from_uci(uci)
            piece = board.piece_at(move.from_square)
            label = _PIECE_LABEL.get(piece.piece_type, "? Other") if piece else "? Other"
        except Exception:
            label = "? Other"
        groups[label].append(uci)

    # Shorter labels and no padding to save tokens
    order = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    lines = []
    for t in order:
        label = _PIECE_LABEL[t]
        if label in groups:
            lines.append(f"{label[0]}: {', '.join(groups[label])}") # E.g. 'K: e1g1, e1f1'
    return "\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# System Prompt — short; tools provide deeper context on demand               #
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """You are {name}, playing as {color}.
Turn: {turn} | FEN: {fen} | Last 3: {history}
Notes: {scratchpad}
{error_context}

Workflow: 1.get_king_safety, 2.get_hanging_pieces, 3.get_all_legal_moves (if needed), 4.update_notes, 5.make_move.

CRITICAL: You MUST strictly execute the 'make_move' tool to finalize your turn. Never just type the move out as text!
"""


# --------------------------------------------------------------------------- #
# ChessAgent                                                                   #
# --------------------------------------------------------------------------- #
class ChessAgent:
    def __init__(self, name: str, color: str, llm):
        self.name   = name
        self.color  = color
        self.llm    = llm
        self.scratchpad: str = "No notes yet."
        self.last_thinking: str = ""

    # ---------------------------------------------------------------------- #
    # Main entry point                                                         #
    # ---------------------------------------------------------------------- #
    def choose_move(
        self,
        fen: str,
        board_ascii: str,
        legal_moves: list[str],
        move_history: list[str],
        on_thinking: Callable[[str], None] | None = None,
        illegal_feedback: str = "",
        attempt: int = 1,
    ) -> tuple[str, str]:

        board = chess.Board(fen)

        # ── Board-query tools ────────────────────────────────────────────── #

        def get_piece_at(square: str) -> str:
            """Return the piece on a square (e.g. 'e4'). Returns 'empty' if none."""
            try:
                sq    = chess.parse_square(square.strip().lower())
                piece = board.piece_at(sq)
                if piece is None:
                    return f"{square}: empty"
                side = "White" if piece.color == chess.WHITE else "Black"
                return f"{square}: {side} {chess.piece_name(piece.piece_type)}"
            except Exception as e:
                return f"Error: {e}"

        def get_attacks_on(square: str) -> str:
            """List which pieces attack a square (e.g. 'e4'). Useful for spotting tactics."""
            try:
                sq = chess.parse_square(square.strip().lower())
                wa = [chess.square_name(s) for s in board.attackers(chess.WHITE, sq)]
                ba = [chess.square_name(s) for s in board.attackers(chess.BLACK, sq)]
                return (
                    f"Attackers of {square} — "
                    f"White: {wa or 'none'}, Black: {ba or 'none'}"
                )
            except Exception as e:
                return f"Error: {e}"

        def get_material_balance(dummy: str = "") -> str:
            """Return material count for both sides. P=1 N=B=3 R=5 Q=9. Call with no argument."""
            v  = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9}
            pm = board.piece_map()
            w  = sum(v.get(p.piece_type, 0) for p in pm.values() if p.color == chess.WHITE)
            b  = sum(v.get(p.piece_type, 0) for p in pm.values() if p.color == chess.BLACK)
            diff = w - b
            adv  = (f"White +{diff}" if diff > 0
                    else f"Black +{-diff}" if diff < 0
                    else "Equal")
            return f"Material — White: {w}, Black: {b}, Balance: {adv}"

        def get_king_safety(dummy: str = "") -> str:
            """Report both kings' positions and whether either is in check. Call with no argument."""
            out = []
            for col, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
                ksq = board.king(col)
                if ksq is None:
                    out.append(f"{name} king: missing")
                    continue
                checked = board.is_check() and board.turn == col
                out.append(f"{name} king @ {chess.square_name(ksq)}"
                            + (" — IN CHECK!" if checked else ""))
            return " | ".join(out)

        def get_hanging_pieces(dummy: str = "") -> str:
            """Find pieces attacked but not defended (hanging). Call with no argument."""
            hanging = []
            for sq, piece in board.piece_map().items():
                if board.attackers(not piece.color, sq) and not board.attackers(piece.color, sq):
                    side = "White" if piece.color == chess.WHITE else "Black"
                    hanging.append(
                        f"{side} {chess.piece_name(piece.piece_type)} @ {chess.square_name(sq)}"
                    )
            return ("Hanging: " + ", ".join(hanging)) if hanging else "No hanging pieces."

        def get_moves_for_piece(piece_name: str) -> str:
            """
            Return all legal moves for a given piece type.
            piece_name: one of 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'
            Example: get_moves_for_piece('knight') → 'g1f3, g1h3, b1c3'
            """
            name_map = {
                "pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
                "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING,
            }
            pt = name_map.get(piece_name.strip().lower())
            if pt is None:
                return f"Unknown piece '{piece_name}'. Use: pawn, knight, bishop, rook, queen, king"
            moves = []
            for uci in legal_moves:
                try:
                    mv    = chess.Move.from_uci(uci)
                    piece = board.piece_at(mv.from_square)
                    if piece and piece.piece_type == pt:
                        moves.append(uci)
                except Exception:
                    pass
            return (f"{piece_name.title()} moves: {', '.join(moves)}"
                    if moves else f"No legal {piece_name} moves available.")

        def get_all_legal_moves(dummy: str = "") -> str:
            """Return all legal moves in UCI format."""
            return f"Legal: {', '.join(legal_moves)}"

        def update_notes(notes: str = "No notes") -> str:
            """Save concise plan (max 150 chars)."""
            self.scratchpad = notes.strip()[:150]
            return "Notes saved."

        def make_move(move: str) -> str:
            """
            Play your chosen move in UCI format (e.g. 'e2e4').
            MUST be an exact string from the 'Legal moves by piece' list.
            Call this ONCE only.
            """
            m = move.strip().lower()
            if m in legal_moves:
                return f"MOVE_ACCEPTED:{m}"
            # Helpful hint: show legal moves from that same source square
            close = [lm for lm in legal_moves if lm.startswith(m[:2])][:8]
            hint  = f" Legal moves from {m[:2]}: {close}" if close else ""
            return f"ILLEGAL: '{m}' is not in the legal moves list.{hint} Try again."

        tools = [
            get_all_legal_moves,
            get_hanging_pieces,
            get_king_safety,
            get_material_balance,
            get_piece_at,
            update_notes,
            make_move,
        ]

        # ── Build prompt ─────────────────────────────────────────────────── #
        history_str = ", ".join(move_history[-3:]) if move_history else "None"
        error_ctx   = (f"ERR: {illegal_feedback}" if illegal_feedback else "")

        sys_prompt = _SYSTEM_PROMPT.format(
            name         = self.name,
            color        = self.color,
            turn         = board.fullmove_number,
            fen          = fen,
            scratchpad   = self.scratchpad,
            history      = history_str,
            error_context= error_ctx,
        )

        # ── Create & invoke agent ─────────────────────────────────────────  #
        try:
            from langchain.agents import create_agent
            graph  = create_agent(self.llm, tools, system_prompt=sys_prompt)
            inputs = {"messages": [{"role": "user", "content": "Analyze the FEN and play your best move by calling make_move."}]}
            result   = graph.invoke(inputs)
            messages = result["messages"]
        except Exception as exc:
            err_log = f"[Agent error] {type(exc).__name__}: {exc}"
            if on_thinking:
                on_thinking(err_log)
            self.last_thinking = err_log
            return "error", err_log

        # ── Extract move — 3-layer priority ───────────────────────────────  #
        #
        # Layer 1 (BEST):  make_move tool returned "MOVE_ACCEPTED:<uci>"
        #                  → the tool itself validated the move, fully reliable
        #
        # Layer 2 (GOOD):  the model's tool_call args for make_move contain a
        #                  valid UCI string (covers cases where tool output is
        #                  missing from result but the call was recorded)
        #
        # Layer 3 (LAST RESORT): regex scan of ONLY the final AI text message
        #                  (NOT the whole log — avoids false matches from FEN
        #                  strings, board notation, or history text)
        #
        chosen_move  = None
        full_log     = ""
        last_ai_text = ""   # only the final AI message content (for regex fallback)

        for msg in messages:
            if msg.type == "human":
                continue

            # ── AI reasoning turn ─────────────────────────────────────────
            if msg.type == "ai":
                if msg.content and msg.content.strip():
                    full_log    += f"🤔 {msg.content.strip()}\n"
                    last_ai_text = msg.content

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc["name"]
                        args = tc.get("args", {})

                        # Pretty arg display
                        if isinstance(args, dict) and len(args) == 1:
                            arg_str = str(list(args.values())[0])
                        elif isinstance(args, dict):
                            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
                        else:
                            arg_str = str(args)

                        full_log += f"🔧 {name}({arg_str})\n"

                        # Layer 2 — extract from make_move call args
                        if name == "make_move" and chosen_move is None:
                            raw = (args.get("move", "") if isinstance(args, dict)
                                   else str(args))
                            m = raw.strip().lower()
                            if m in legal_moves:
                                chosen_move = m

            # ── Tool result ───────────────────────────────────────────────
            elif msg.type == "tool" and msg.content:
                content = msg.content.strip()

                # Layer 1 — MOVE_ACCEPTED marker
                if content.startswith("MOVE_ACCEPTED:"):
                    move_val    = content.split("MOVE_ACCEPTED:", 1)[1].strip()
                    chosen_move = move_val
                    full_log   += f"✅ Move accepted: {move_val}\n"
                elif content.startswith("ILLEGAL:"):
                    full_log += f"⛔ {content}\n"
                else:
                    full_log += f"📋 {content}\n"

        full_log += "\n" + "─" * 36 + "\n"

        # Layer 3 — regex scan of LAST AI message only (avoids FEN false-positives)
        if not chosen_move and last_ai_text:
            for m in re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', last_ai_text, re.I):
                if m.lower() in legal_moves:
                    chosen_move = m.lower()
                    full_log   += f"🔍 Layer 3: Extracted from reasoning text: {chosen_move}\n"
                    break

        if not chosen_move:
            full_log += "❌ No valid move found in this turn. Triggering engine fallback... 🔀\n"
        else:
            # Show which layer actually caught the move for debugging
            if "accepted" in full_log:
                pass # Layer 1 already logged
            elif "Action" in full_log:
                 full_log += f"✅ Layer 2: Captured from tool arguments: {chosen_move}\n"

        if on_thinking:
            on_thinking(full_log)

        self.last_thinking = full_log
        return (chosen_move or "error", full_log)

    def fallback_move(self, legal_moves: list[str]) -> str:
        """Return the first legal move as a guaranteed fallback (never forfeit)."""
        return legal_moves[0] if legal_moves else "error"

    def reset(self):
        self.scratchpad    = "No notes yet."
        self.last_thinking = ""