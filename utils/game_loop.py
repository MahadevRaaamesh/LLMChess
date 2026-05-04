"""
utils/game_loop.py
Orchestrates the game: alternates agents, validates moves,
handles retries, and fires callbacks the UI listens to.
"""
from __future__ import annotations
import threading
import time
from typing import Callable

from chess_engine import ChessEngine
from agents import ChessAgent


class GameLoop:
    def __init__(
        self,
        engine: ChessEngine,
        white_agent: ChessAgent,
        black_agent: ChessAgent,
        # Callbacks (all called from the game thread — UI must use after())
        on_board_update: Callable[[], None] | None = None,
        on_agent_thinking: Callable[[str, str], None] | None = None,
        on_move_made: Callable[[str, str], None] | None = None,
        on_game_over: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        # on_retry(color, attempt, white_fails, black_fails, fallbacks)
        on_retry: Callable[[str, int, int, int, int], None] | None = None,
        move_delay: float = 5.0,
        retry_delay: float = 4.0,   # seconds to wait between retry attempts
    ):
        self.engine        = engine
        self.agents        = {
            "White": white_agent,
            "Black": black_agent,
        }
        self.on_board_update   = on_board_update   or (lambda: None)
        self.on_agent_thinking = on_agent_thinking or (lambda c, t: None)
        self.on_move_made      = on_move_made      or (lambda c, m: None)
        self.on_game_over      = on_game_over      or (lambda r: None)
        self.on_status         = on_status         or (lambda s: None)
        self.on_retry          = on_retry          or (lambda c, a, wf, bf, fb: None)
        self.move_delay        = move_delay
        self.retry_delay       = retry_delay

        # Fail counters (visible in UI)
        self.fails     = {"White": 0, "Black": 0}
        self.fallbacks = 0

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()   # not paused by default

    # ------------------------------------------------------------------ #
    # Control                                                              #
    # ------------------------------------------------------------------ #
    def start(self):
        self._stop_event.clear()
        self._pause_event.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()   # unblock if paused

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    def _run(self):
        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            if self.engine.is_game_over():
                result = self.engine.game_result()
                self.on_game_over(result)
                break

            color   = self.engine.whose_turn()
            agent   = self.agents[color]
            fen     = self.engine.get_fen()
            legal   = self.engine.get_legal_moves()
            history = self.engine.get_move_history_san()

            self.on_status(f"{color} is thinking…")

            illegal_feedback = ""
            move_made = False

            for attempt in range(1, 6):   # up to 5 LLM attempts before fallback
                try:
                    def _thinking_cb(text, _color=color):
                        self.on_agent_thinking(_color, text)

                    board_ascii = str(self.engine.board)

                    uci_move, raw = agent.choose_move(
                        fen=fen,
                        board_ascii=board_ascii,
                        legal_moves=legal,
                        move_history=history,
                        on_thinking=_thinking_cb,
                        illegal_feedback=illegal_feedback,
                        attempt=attempt,
                    )
                except Exception as e:
                    self.fails[color] += 1
                    self.on_status(f"{color} error attempt {attempt}/5: {e}")
                    self.on_agent_thinking(color, f"[ERROR attempt {attempt}/5] {e}")
                    self.on_retry(color, attempt,
                                  self.fails["White"], self.fails["Black"], self.fallbacks)
                    illegal_feedback = str(e)
                    if attempt < 5:
                        time.sleep(self.retry_delay)   # back off before retry
                    continue

                ok, msg = self.engine.try_move(uci_move)
                if ok:
                    self.on_move_made(color, uci_move)
                    self.on_board_update()
                    self.on_status(f"{color} played {uci_move}")
                    move_made = True
                    break
                else:
                    self.fails[color] += 1
                    self.on_status(
                        f"{color} illegal move (attempt {attempt}/5): {uci_move}")
                    illegal_feedback = msg
                    self.on_retry(color, attempt,
                                  self.fails["White"], self.fails["Black"], self.fallbacks)
                    if attempt < 5:
                        time.sleep(self.retry_delay)   # back off before retry

            if not move_made:
                # Ultimate fallback: play the first legal move rather than forfeiting
                fallback = agent.fallback_move(legal)
                if fallback != "error":
                    ok, _ = self.engine.try_move(fallback)
                    if ok:
                        self.fallbacks += 1
                        self.on_agent_thinking(
                            color,
                            f"[Fallback #{self.fallbacks}] LLM failed 5 times — "
                            f"playing {fallback} automatically."
                        )
                        self.on_move_made(color, fallback)
                        self.on_board_update()
                        self.on_status(f"{color} played {fallback} (auto-fallback #{self.fallbacks})")
                        self.on_retry(color, 5,
                                      self.fails["White"], self.fails["Black"], self.fallbacks)
                        move_made = True

            if not move_made:
                self.on_game_over(
                    f"{color} forfeited — no legal move found after 5 attempts")
                break

            time.sleep(self.move_delay)