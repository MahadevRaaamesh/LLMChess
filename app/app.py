"""
ui/app.py
Tkinter UI for ChessLLM.

Layout
------
┌─────────────────────────────────────────────────────────────┐
│  White Agent Thinking          Black Agent Thinking          │
│  (scrolled text, left)         (scrolled text, right)        │
├──────────────────────┬──────────────────────────────────────┤
│                      │  Move History + Status bar            │
│   Chess Board        │                                       │
│   (Canvas 480×480)   │                                       │
│                      │                                       │
├──────────────────────┴──────────────────────────────────────┤
│  [Start]  [Pause/Resume]  [Stop]  [New Game]   speed slider │
└─────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations
import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog

# Make sure project root is on path when running ui/app.py directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chess_engine import ChessEngine
from agents import ChessAgent, get_llm
from utils import GameLoop

# ── Colours ────────────────────────────────────────────────────────────────
LIGHT_SQ   = "#F0D9B5"
DARK_SQ    = "#B58863"
HIGHLIGHT  = "#AAD751"
BG         = "#1E1E2E"
PANEL_BG   = "#2A2A3E"
TEXT_FG    = "#CDD6F4"
WHITE_CLR  = "#FFFFFF"
BLACK_CLR  = "#000000"
ACCENT     = "#89B4FA"

PIECE_UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

SQ = 60          # pixels per square
BOARD_PX = SQ * 8


# ── Setup dialog ────────────────────────────────────────────────────────── #
class SetupDialog(tk.Toplevel):
    """Modal dialog to collect provider / API key / model before the game."""

    PROVIDERS   = ["huggingface", "novita", "featherless", "openai", "anthropic", "google", "groq", "deepseek", "perplexity", "ollama", "grok", "lm-studio", "vllm", "llama-cpp"]
    RETRY_DELAY = 4.0   # seconds between retry attempts
    DEFAULTS  = {
        "huggingface": "openai-community/gpt2",
        "novita":      "meta-llama/Llama-3.2-1B-Instruct",
        "featherless": "Qwen/Qwen2.5-1.5B-Instruct",
        "openai":      "gpt-4o",
        "anthropic":   "claude-3-5-sonnet-20241022",
        "google":      "gemini-1.5-flash",
        "groq":        "llama-3.1-70b-versatile",
        "deepseek":    "deepseek-chat",
        "perplexity":  "sonar-pro",
        "ollama":      "llama3",
        "grok":        "grok-beta",
        "lm-studio":   "model-identifier",
        "vllm":        "model-identifier",
        "llama-cpp":   "local-model",
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.title("ChessLLM — Setup")
        self.resizable(False, False)
        self.grab_set()

        self.result = None   # set to dict on OK

        self._build()
        self.wait_window()

    def _build(self):
        pad = {"padx": 10, "pady": 6}

        # ── White Agent Frame ──────────────────────────────────────────
        w_frame = tk.LabelFrame(self, text="⬜ White Agent Settings", font=("Helvetica", 10, "bold"))
        w_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=6)
        self.w_vars = self._add_agent_fields(w_frame)

        # ── Black Agent Frame ──────────────────────────────────────────
        b_frame = tk.LabelFrame(self, text="⬛ Black Agent Settings", font=("Helvetica", 10, "bold"))
        b_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        self.b_vars = self._add_agent_fields(b_frame)

        # ── Global Settings ─────────────────────────────────────────────
        g_frame = tk.Frame(self)
        g_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=6)

        tk.Label(g_frame, text="Move delay (s):").pack(side="left", padx=5)
        self.delay_var = tk.DoubleVar(value=5.0)
        tk.Spinbox(g_frame, from_=0.5, to=30.0, increment=0.5,
                   textvariable=self.delay_var, width=8).pack(side="left", padx=5)

        # ── Buttons ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=3, column=0, pady=10)
        tk.Button(btn_frame, text="Start Game", command=self._ok,
                  width=14, bg=ACCENT).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", command=self.destroy,
                  width=10).pack(side="left", padx=6)

    def _add_agent_fields(self, frame):
        pad = {"padx": 5, "pady": 4}
        
        vars = {
            "provider": tk.StringVar(value="huggingface"),
            "key":      tk.StringVar(),
            "model":    tk.StringVar(value=self.DEFAULTS["huggingface"]),
            "entry":    None
        }

        tk.Label(frame, text="Provider:").grid(row=0, column=0, sticky="w", **pad)
        cb = ttk.Combobox(frame, textvariable=vars["provider"],
                          values=self.PROVIDERS, state="readonly", width=15)
        cb.grid(row=0, column=1, sticky="ew", **pad)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_provider_change(vars))

        tk.Label(frame, text="Key/URL:").grid(row=1, column=0, sticky="w", **pad)
        vars["entry"] = tk.Entry(frame, textvariable=vars["key"], show="*", width=35)
        vars["entry"].grid(row=1, column=1, sticky="ew", **pad)

        tk.Label(frame, text="Model:").grid(row=2, column=0, sticky="w", **pad)
        tk.Entry(frame, textvariable=vars["model"], width=35).grid(
            row=2, column=1, sticky="ew", **pad)

        return vars

    def _on_provider_change(self, vars):
        p = vars["provider"].get()
        vars["model"].set(self.DEFAULTS.get(p, ""))
        
        if p in ["llama-cpp", "lm-studio", "vllm"]:
            vars["entry"].configure(show="")
            if not vars["key"].get() or vars["key"].get().startswith("http"):
                if p == "llama-cpp":
                    vars["key"].set("http://localhost:8080/v1")
                elif p == "lm-studio":
                    vars["key"].set("http://localhost:1234/v1")
                elif p == "vllm":
                    vars["key"].set("http://localhost:8000/v1")
        else:
            vars["entry"].configure(show="*")
            if vars["key"].get() in ["http://localhost:8080/v1", "http://localhost:1234/v1", "http://localhost:8000/v1"]:
                vars["key"].set("")

    def _ok(self):
        self.result = {
            "white": {
                "provider": self.w_vars["provider"].get(),
                "api_key":  self.w_vars["key"].get().strip(),
                "model":    self.w_vars["model"].get().strip(),
            },
            "black": {
                "provider": self.b_vars["provider"].get(),
                "api_key":  self.b_vars["key"].get().strip(),
                "model":    self.b_vars["model"].get().strip(),
            },
            "delay": self.delay_var.get(),
        }
        self.destroy()


# ── Main application ─────────────────────────────────────────────────────── #
class ChessLLMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChessLLM — LLM vs LLM")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.engine: ChessEngine | None = None
        self.game_loop: GameLoop | None = None
        self._paused = False

        self._build_ui()
        self.after(100, self._show_setup)

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # ── top: two agent thinking panels ──────────────────────────────
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=8, pady=(8, 0))

        self.white_panel = self._make_thinking_panel(top, "⬜ White Agent Thinking", "#E8E8E8")
        self.white_panel.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.black_panel = self._make_thinking_panel(top, "⬛ Black Agent Thinking", "#888888")
        self.black_panel.pack(side="right", fill="both", expand=True, padx=(4, 0))

        # ── middle: board + move history ────────────────────────────────
        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="both", expand=True, padx=8, pady=8)

        # Board canvas
        board_frame = tk.Frame(mid, bg=BG)
        board_frame.pack(side="left")

        self.canvas = tk.Canvas(board_frame, width=BOARD_PX, height=BOARD_PX,
                                bg=BG, highlightthickness=0)
        self.canvas.pack()
        self._draw_empty_board()

        # Right panel: move history + status
        right = tk.Frame(mid, bg=PANEL_BG, bd=1, relief="flat")
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        tk.Label(right, text="Move History", bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica", 11, "bold")).pack(pady=(6, 2))

        self.history_box = scrolledtext.ScrolledText(
            right, bg=PANEL_BG, fg=TEXT_FG, font=("Courier", 10),
            state="disabled", wrap="word", height=14)
        self.history_box.pack(fill="both", expand=True, padx=6, pady=4)

        # ── Fail / Fallback stats bar ────────────────────────────────────
        stats_frame = tk.Frame(right, bg=PANEL_BG)
        stats_frame.pack(fill="x", padx=6, pady=(0, 2))

        self.white_fails_var    = tk.StringVar(value="⬜ Fails: 0")
        self.black_fails_var    = tk.StringVar(value="⬛ Fails: 0")
        self.fallbacks_var      = tk.StringVar(value="🔀 Fallbacks: 0")

        lbl_cfg = dict(bg=PANEL_BG, font=("Courier", 9))
        tk.Label(stats_frame, textvariable=self.white_fails_var,
                 fg="#E8E8E8", **lbl_cfg).pack(side="left", padx=(0, 8))
        tk.Label(stats_frame, textvariable=self.black_fails_var,
                 fg="#888888", **lbl_cfg).pack(side="left", padx=(0, 8))
        tk.Label(stats_frame, textvariable=self.fallbacks_var,
                 fg="#F38BA8", **lbl_cfg).pack(side="left")

        self.status_var = tk.StringVar(value="Configure a game to begin.")
        tk.Label(right, textvariable=self.status_var, bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica", 10, "italic"), wraplength=220).pack(pady=(2, 6))

        # ── bottom: control bar ─────────────────────────────────────────
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=8, pady=(0, 8))

        btn_cfg = dict(bg=ACCENT, fg=BLACK_CLR, font=("Helvetica", 10, "bold"),
                       relief="flat", padx=10, pady=4, cursor="hand2")

        self.btn_start = tk.Button(bar, text="▶ Start", command=self._show_setup, **btn_cfg)
        self.btn_start.pack(side="left", padx=4)

        self.btn_pause = tk.Button(bar, text="⏸ Pause", command=self._toggle_pause,
                                   state="disabled", **btn_cfg)
        self.btn_pause.pack(side="left", padx=4)

        self.btn_stop = tk.Button(bar, text="⏹ Stop", command=self._stop_game,
                                  state="disabled", **btn_cfg)
        self.btn_stop.pack(side="left", padx=4)

        self.btn_new = tk.Button(bar, text="🔄 New Game", command=self._show_setup, **btn_cfg)
        self.btn_new.pack(side="left", padx=4)

        tk.Label(bar, text="Speed:", bg=BG, fg=TEXT_FG).pack(side="left", padx=(16, 2))
        self.speed_var = tk.DoubleVar(value=5.0)
        self.speed_slider = ttk.Scale(bar, from_=0.5, to=30.0,
                                      variable=self.speed_var, orient="horizontal",
                                      length=140)
        self.speed_slider.pack(side="left")
        tk.Label(bar, textvariable=self.speed_var, bg=BG, fg=TEXT_FG,
                 width=4).pack(side="left")

    def _make_thinking_panel(self, parent, title: str, label_color: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="flat")
        tk.Label(frame, text=title, bg=PANEL_BG, fg=label_color,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=6, pady=(4, 0))
        box = scrolledtext.ScrolledText(
            frame, bg=PANEL_BG, fg=TEXT_FG, font=("Courier", 9),
            state="disabled", wrap="word", height=8)
        box.pack(fill="both", expand=True, padx=6, pady=(2, 6))
        frame._textbox = box   # attach for easy access
        return frame

    # ------------------------------------------------------------------ #
    # Board drawing                                                        #
    # ------------------------------------------------------------------ #
    def _draw_empty_board(self):
        self.canvas.delete("all")
        for r in range(8):
            for c in range(8):
                color = LIGHT_SQ if (r + c) % 2 == 0 else DARK_SQ
                x0, y0 = c * SQ, r * SQ
                self.canvas.create_rectangle(x0, y0, x0 + SQ, y0 + SQ,
                                             fill=color, outline="")
        # Rank / file labels
        files = "abcdefgh"
        for i in range(8):
            self.canvas.create_text(i * SQ + SQ // 2, BOARD_PX - 4,
                                    text=files[i], fill="#555", font=("Helvetica", 7))
            self.canvas.create_text(4, i * SQ + SQ // 2,
                                    text=str(8 - i), fill="#555", font=("Helvetica", 7))

    def _render_board(self):
        if not self.engine:
            return
        board = self.engine.board
        last_move = self.engine.raw_history[-1] if self.engine.raw_history else None

        self.canvas.delete("all")
        for r in range(8):
            for c in range(8):
                sq_index = (7 - r) * 8 + c   # chess.Square: a1=0 … h8=63
                is_light = (r + c) % 2 == 0
                color = LIGHT_SQ if is_light else DARK_SQ

                # Highlight last move
                import chess
                sq = chess.square(c, 7 - r)
                if last_move and sq in (last_move.from_square, last_move.to_square):
                    color = HIGHLIGHT

                x0, y0 = c * SQ, r * SQ
                self.canvas.create_rectangle(x0, y0, x0 + SQ, y0 + SQ,
                                             fill=color, outline="")

                piece = board.piece_at(sq)
                if piece:
                    symbol = PIECE_UNICODE.get(piece.symbol(), "?")
                    fg = WHITE_CLR if piece.color else BLACK_CLR
                    # Shadow for readability
                    self.canvas.create_text(
                        x0 + SQ // 2 + 1, y0 + SQ // 2 + 1,
                        text=symbol, font=("Segoe UI Symbol", 28), fill="#333")
                    self.canvas.create_text(
                        x0 + SQ // 2, y0 + SQ // 2,
                        text=symbol, font=("Segoe UI Symbol", 28), fill=fg)

        # Labels
        files = "abcdefgh"
        for i in range(8):
            self.canvas.create_text(i * SQ + SQ // 2, BOARD_PX - 5,
                                    text=files[i], fill="#555", font=("Helvetica", 7))
            self.canvas.create_text(5, i * SQ + SQ // 2,
                                    text=str(8 - i), fill="#555", font=("Helvetica", 7))

    # ------------------------------------------------------------------ #
    # Helpers: write to panels                                             #
    # ------------------------------------------------------------------ #
    def _append_thinking(self, color: str, text: str):
        panel = self.white_panel if color == "White" else self.black_panel
        box: scrolledtext.ScrolledText = panel._textbox
        box.configure(state="normal")
        # box.delete("1.0", "end")  # Remove this to keep a continuous log
        box.insert("end", "\n" + "-"*40 + "\n")
        box.insert("end", text + "\n")
        box.see("end")
        box.configure(state="disabled")

    def _append_history(self, text: str):
        self.history_box.configure(state="normal")
        self.history_box.insert("end", text + "\n")
        self.history_box.see("end")
        self.history_box.configure(state="disabled")

    def _clear_history(self):
        self.history_box.configure(state="normal")
        self.history_box.delete("1.0", "end")
        self.history_box.configure(state="disabled")

    # ------------------------------------------------------------------ #
    # Game setup & control                                                 #
    # ------------------------------------------------------------------ #
    def _show_setup(self):
        if self.game_loop and self.game_loop.is_running():
            self._stop_game()

        dlg = SetupDialog(self)
        cfg = dlg.result
        if not cfg:
            return

        try:
            # Initialize separate LLMs for White and Black
            white_llm = get_llm(
                cfg["white"]["provider"], 
                cfg["white"]["api_key"], 
                cfg["white"]["model"]
            )
            black_llm = get_llm(
                cfg["black"]["provider"], 
                cfg["black"]["api_key"], 
                cfg["black"]["model"]
            )
        except Exception as e:
            messagebox.showerror("LLM Error", f"Failed to initialize LLMs:\n{e}")
            return

        self.engine = ChessEngine()
        white_agent = ChessAgent("Agent White", "White", white_llm)
        black_agent = ChessAgent("Agent Black", "Black", black_llm)

        self.speed_var.set(cfg["delay"])

        self._clear_history()
        self.white_panel._textbox.configure(state="normal")
        self.white_panel._textbox.delete("1.0", "end")
        self.white_panel._textbox.configure(state="disabled")
        self.black_panel._textbox.configure(state="normal")
        self.black_panel._textbox.delete("1.0", "end")
        self.black_panel._textbox.configure(state="disabled")

        # Reset fail stats
        self.white_fails_var.set("⬜ Fails: 0")
        self.black_fails_var.set("⬛ Fails: 0")
        self.fallbacks_var.set("🔀 Fallbacks: 0")

        self._render_board()
        self.status_var.set("Game starting…")

        self.game_loop = GameLoop(
            engine=self.engine,
            white_agent=white_agent,
            black_agent=black_agent,
            on_board_update=lambda: self.after(0, self._render_board),
            on_agent_thinking=lambda c, t: self.after(0, self._append_thinking, c, t),
            on_move_made=lambda c, m: self.after(
                0, self._append_history,
                f"{'  ' if c=='Black' else ''}{len(self.engine.move_history)}. {m}"
            ),
            on_game_over=lambda r: self.after(0, self._on_game_over, r),
            on_status=lambda s: self.after(0, self.status_var.set, s),
            on_retry=lambda c, a, wf, bf, fb: self.after(
                0, self._on_retry, c, a, wf, bf, fb),
            move_delay=cfg["delay"],
            retry_delay=SetupDialog.RETRY_DELAY,
        )

        # Keep move_delay in sync with slider
        def _watch_speed():
            if self.game_loop:
                self.game_loop.move_delay = self.speed_var.get()
            self.after(500, _watch_speed)
        _watch_speed()

        self.btn_pause.configure(state="normal")
        self.btn_stop.configure(state="normal")
        self.btn_start.configure(state="disabled")

        self.game_loop.start()

    def _toggle_pause(self):
        if not self.game_loop:
            return
        if self._paused:
            self.game_loop.resume()
            self.btn_pause.configure(text="⏸ Pause")
            self._paused = False
        else:
            self.game_loop.pause()
            self.btn_pause.configure(text="▶ Resume")
            self._paused = True

    def _stop_game(self):
        if self.game_loop:
            self.game_loop.stop()
        self.btn_pause.configure(state="disabled", text="⏸ Pause")
        self.btn_stop.configure(state="disabled")
        self.btn_start.configure(state="normal")
        self._paused = False
        self.status_var.set("Game stopped.")

    def _on_retry(self, color: str, attempt: int, white_fails: int, black_fails: int, fallbacks: int):
        """Update the live fail-stats bar whenever a retry or fallback happens."""
        self.white_fails_var.set(f"⬜ Fails: {white_fails}")
        self.black_fails_var.set(f"⬛ Fails: {black_fails}")
        self.fallbacks_var.set(f"🔀 Fallbacks: {fallbacks}")
        # Also log to the move history so you can see when it happened
        self._append_history(
            f"  ⚠️  {color} retry {attempt}/5 — "
            f"W:{white_fails} B:{black_fails} FB:{fallbacks}"
        )

    def _on_game_over(self, result: str):
        self.status_var.set(f"Game over: {result}")
        self.btn_pause.configure(state="disabled")
        self.btn_stop.configure(state="disabled")
        self.btn_start.configure(state="normal")
        messagebox.showinfo("Game Over", result)


# ── Entry point ──────────────────────────────────────────────────────────── #
def run():
    app = ChessLLMApp()
    app.mainloop()


if __name__ == "__main__":
    run()