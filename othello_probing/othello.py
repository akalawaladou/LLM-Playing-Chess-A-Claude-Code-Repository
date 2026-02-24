"""
Othello (Reversi) game engine.

Implements the full game logic needed to:
- Generate random legal games (training data for the GPT)
- Track ground-truth board state at each move (labels for probing)

Board encoding:
    0 = empty, 1 = black, -1 = white
    Moves are integers 0..63 mapping to row*8 + col.
"""

import numpy as np
from typing import List, Tuple, Optional

# 8 directions: (drow, dcol)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


class OthelloGame:
    """Manages one Othello game."""

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        # Standard starting position
        self.board[3, 3] = -1  # white
        self.board[3, 4] = 1   # black
        self.board[4, 3] = 1   # black
        self.board[4, 4] = -1  # white
        self.current_player = 1  # black starts
        self.move_history: List[int] = []
        self.board_states: List[np.ndarray] = [self.board.copy()]

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < 8 and 0 <= c < 8

    def _flips_in_direction(self, r: int, c: int, dr: int, dc: int,
                            player: int) -> List[Tuple[int, int]]:
        """Return list of opponent pieces that would be flipped in one direction."""
        flips = []
        nr, nc = r + dr, c + dc
        while self._in_bounds(nr, nc) and self.board[nr, nc] == -player:
            flips.append((nr, nc))
            nr += dr
            nc += dc
        # Must end on a friendly piece to actually capture
        if flips and self._in_bounds(nr, nc) and self.board[nr, nc] == player:
            return flips
        return []

    def get_legal_moves(self, player: Optional[int] = None) -> List[int]:
        """Return list of legal move indices (0..63) for `player`."""
        if player is None:
            player = self.current_player
        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r, c] != 0:
                    continue
                for dr, dc in DIRECTIONS:
                    if self._flips_in_direction(r, c, dr, dc, player):
                        moves.append(r * 8 + c)
                        break
        return moves

    def play_move(self, move: int) -> bool:
        """Play `move` (0..63) for current player. Returns True if successful."""
        r, c = divmod(move, 8)
        if self.board[r, c] != 0:
            return False

        all_flips = []
        for dr, dc in DIRECTIONS:
            all_flips.extend(
                self._flips_in_direction(r, c, dr, dc, self.current_player)
            )

        if not all_flips:
            return False

        # Place piece and flip captured pieces
        self.board[r, c] = self.current_player
        for fr, fc in all_flips:
            self.board[fr, fc] = self.current_player

        self.move_history.append(move)
        self.board_states.append(self.board.copy())
        self.current_player *= -1  # switch player
        return True

    def is_game_over(self) -> bool:
        if self.get_legal_moves(1) or self.get_legal_moves(-1):
            return False
        return True


def generate_random_game(rng: np.random.Generator) -> Tuple[List[int], List[np.ndarray]]:
    """
    Play a random legal game of Othello.

    Returns:
        moves: list of move indices (length = number of moves)
        boards: list of board states *after* each move (same length)
    """
    game = OthelloGame()
    passes = 0

    while not game.is_game_over() and passes < 2:
        legal = game.get_legal_moves()
        if not legal:
            game.current_player *= -1  # pass
            passes += 1
            continue
        passes = 0
        move = legal[rng.integers(len(legal))]
        game.play_move(move)

    return game.move_history, game.board_states[1:]  # skip initial state


def generate_dataset(n_games: int, seed: int = 42) -> Tuple[List[List[int]], List[List[np.ndarray]]]:
    """Generate `n_games` random Othello games."""
    rng = np.random.default_rng(seed)
    all_moves = []
    all_boards = []
    for _ in range(n_games):
        moves, boards = generate_random_game(rng)
        if len(moves) > 0:
            all_moves.append(moves)
            all_boards.append(boards)
    return all_moves, all_boards


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    moves, boards = generate_dataset(5, seed=0)
    for i, (m, b) in enumerate(zip(moves, boards)):
        print(f"Game {i}: {len(m)} moves")
        # Show final board
        symbols = {0: ".", 1: "B", -1: "W"}
        final = b[-1]
        for r in range(8):
            print(" ".join(symbols[final[r, c]] for c in range(8)))
        print()
