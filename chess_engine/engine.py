import chess

class ChessEngine:
    def __init__(self):
        self.board = chess.Board()

    @property
    def raw_history(self):
        """Returns the stack of moves as chess.Move objects."""
        return list(self.board.move_stack)

    @property
    def move_history(self):
        """Returns the stack of moves as SAN strings."""
        return self.get_move_history_san()

    def get_fen(self):
        return self.board.fen()

    def get_legal_moves(self):
        """Returns legal moves in UCI format as expected by the agent."""
        return [move.uci() for move in self.board.legal_moves]

    def whose_turn(self):
        return "White" if self.board.turn == chess.WHITE else "Black"

    def get_move_history_san(self):
        """Reconstructs the move history in SAN format."""
        temp_board = chess.Board()
        history = []
        for move in self.board.move_stack:
            history.append(temp_board.san(move))
            temp_board.push(move)
        return history

    def try_move(self, move_uci):
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True, ""
            else:
                return False, f"Move {move_uci} is not legal in this position."
        except ValueError as e:
            return False, f"Invalid move format: {str(e)}"

    def is_game_over(self):
        return self.board.is_game_over()

    def game_result(self):
        if self.board.is_checkmate():
            return "Checkmate"
        if self.board.is_stalemate():
            return "Stalemate"
        if self.board.is_insufficient_material():
            return "Insufficient Material"
        if self.board.is_seventyfive_moves():
            return "75-move rule"
        if self.board.is_fivefold_repetition():
            return "Fivefold repetition"
        return "Ongoing"

    def reset(self):
        self.board.reset()
