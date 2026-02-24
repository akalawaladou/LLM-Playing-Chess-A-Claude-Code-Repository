"""
Chess game engine — pure Python + NumPy (no external chess library).

Board encoding (8x8 np.int8):
    0  = empty
    1  = white Pawn    7  = black Pawn
    2  = white Knight  8  = black Knight
    3  = white Bishop  9  = black Bishop
    4  = white Rook    10 = black Rook
    5  = white Queen   11 = black Queen
    6  = white King    12 = black King

White plays from rows 6-7 (ranks 1-2), black from rows 0-1 (ranks 7-8).
Moves are strings in UCI format: "e2e4", "g1f3", "e7e8q" (promotion), etc.
"""

import json
import os
import numpy as np
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WHITE, BLACK = 1, -1  # side to move

# Piece type indices (piece // 1 = type, offset 0 for white, 6 for black)
EMPTY = 0
W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING = 1, 2, 3, 4, 5, 6
B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING = 7, 8, 9, 10, 11, 12

PIECE_SYMBOLS = {
    0: ".", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
    7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
}

FILES = "abcdefgh"
RANKS = "87654321"  # row 0 = rank 8

def sq_to_uci(r: int, c: int) -> str:
    return FILES[c] + RANKS[r]

def uci_to_sq(s: str) -> Tuple[int, int]:
    return (RANKS.index(s[1]), FILES.index(s[0]))


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def is_white_piece(p: int) -> bool: return 1 <= p <= 6
def is_black_piece(p: int) -> bool: return 7 <= p <= 12
def piece_color(p: int) -> Optional[int]:
    if is_white_piece(p): return WHITE
    if is_black_piece(p): return BLACK
    return None

def piece_type(p: int) -> int:
    """Return 1-6 regardless of color."""
    if 7 <= p <= 12: return p - 6
    return p

def make_piece(ptype: int, color: int) -> int:
    """ptype in 1-6, color in {WHITE, BLACK}."""
    return ptype if color == WHITE else ptype + 6

def initial_board() -> np.ndarray:
    b = np.zeros((8, 8), dtype=np.int8)
    back_row = [W_ROOK, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING,
                W_BISHOP, W_KNIGHT, W_ROOK]
    for c, p in enumerate(back_row):
        b[7, c] = p              # white back rank (row 7)
        b[0, c] = p + 6          # black back rank (row 0)
        b[6, c] = W_PAWN         # white pawns (row 6)
        b[1, c] = B_PAWN         # black pawns (row 1)
    return b


# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------

def _sliding_moves(board, r, c, directions):
    """Generate moves for rook/bishop/queen."""
    color = piece_color(board[r, c])
    moves = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            target = board[nr, nc]
            if target == EMPTY:
                moves.append((nr, nc))
            else:
                if piece_color(target) != color:
                    moves.append((nr, nc))  # capture
                break
            nr += dr
            nc += dc
    return moves


def _king_moves(board, r, c):
    color = piece_color(board[r, c])
    moves = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == dc == 0: continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if piece_color(board[nr, nc]) != color:
                    moves.append((nr, nc))
    return moves


def _pseudo_legal_moves(board, r, c, en_passant_sq, castling_rights):
    """Return list of (to_r, to_c, promo) without checking for king safety."""
    p = board[r, c]
    if p == EMPTY:
        return []
    color = piece_color(p)
    ptype = piece_type(p)
    moves = []

    # ---- Pawn ----
    if ptype == 1:
        direction = -1 if color == WHITE else 1
        start_row = 6 if color == WHITE else 1
        promo_row = 0 if color == WHITE else 7

        # Forward one
        nr = r + direction
        if 0 <= nr < 8 and board[nr, c] == EMPTY:
            if nr == promo_row:
                for promo in ["q", "r", "b", "n"]:
                    moves.append((nr, c, promo))
            else:
                moves.append((nr, c, ""))
            # Double advance from start
            if r == start_row:
                nr2 = r + 2 * direction
                if board[nr2, c] == EMPTY:
                    moves.append((nr2, c, ""))

        # Captures (diagonal)
        for dc in [-1, 1]:
            nc = c + dc
            if 0 <= nc < 8 and 0 <= nr < 8:
                target = board[nr, nc]
                if target != EMPTY and piece_color(target) != color:
                    if nr == promo_row:
                        for promo in ["q", "r", "b", "n"]:
                            moves.append((nr, nc, promo))
                    else:
                        moves.append((nr, nc, ""))
                # En passant
                elif en_passant_sq and (nr, nc) == en_passant_sq:
                    moves.append((nr, nc, "ep"))
        return moves

    # ---- Knight ----
    if ptype == 2:
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if piece_color(board[nr, nc]) != color:
                    moves.append((nr, nc, ""))
        return moves

    # ---- Bishop ----
    if ptype == 3:
        for sq in _sliding_moves(board, r, c, [(-1,-1),(-1,1),(1,-1),(1,1)]):
            moves.append((sq[0], sq[1], ""))
        return moves

    # ---- Rook ----
    if ptype == 4:
        for sq in _sliding_moves(board, r, c, [(-1,0),(1,0),(0,-1),(0,1)]):
            moves.append((sq[0], sq[1], ""))
        return moves

    # ---- Queen ----
    if ptype == 5:
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for sq in _sliding_moves(board, r, c, dirs):
            moves.append((sq[0], sq[1], ""))
        return moves

    # ---- King ----
    if ptype == 6:
        for sq in _king_moves(board, r, c):
            moves.append((sq[0], sq[1], ""))
        # Castling
        back_row = 7 if color == WHITE else 0
        if r == back_row and c == 4:
            side = "w" if color == WHITE else "b"
            # Kingside
            if castling_rights.get(side + "K") and \
               board[back_row, 5] == EMPTY and board[back_row, 6] == EMPTY:
                moves.append((back_row, 6, "castle"))
            # Queenside
            if castling_rights.get(side + "Q") and \
               board[back_row, 3] == EMPTY and board[back_row, 2] == EMPTY \
               and board[back_row, 1] == EMPTY:
                moves.append((back_row, 2, "castle"))
        return moves

    return moves


def _is_in_check(board, color):
    """Check whether `color` king is attacked by any opponent piece."""
    king_piece = W_KING if color == WHITE else B_KING
    pos = np.argwhere(board == king_piece)
    if len(pos) == 0:
        return True  # king missing = in check (shouldn't happen normally)
    kr, kc = pos[0]

    opp = BLACK if color == WHITE else WHITE
    for r in range(8):
        for c in range(8):
            if piece_color(board[r, c]) == opp:
                for mv in _pseudo_legal_moves(board, r, c, None, {}):
                    if mv[0] == kr and mv[1] == kc:
                        return True
    return False


def _apply_move(board, r, c, nr, nc, special, color):
    """Return new board after applying move. Does not modify original."""
    b = board.copy()
    p = b[r, c]
    b[r, c] = EMPTY

    if special == "ep":
        # Capture the pawn behind
        cap_row = nr + (1 if color == WHITE else -1)
        b[cap_row, nc] = EMPTY
    elif special == "castle":
        # Move rook too
        back_row = 7 if color == WHITE else 0
        if nc == 6:  # kingside
            b[back_row, 5] = b[back_row, 7]
            b[back_row, 7] = EMPTY
        else:  # queenside
            b[back_row, 3] = b[back_row, 0]
            b[back_row, 0] = EMPTY
    elif special in ("q", "r", "b", "n"):
        ptype = {"q": 5, "r": 4, "b": 3, "n": 2}[special]
        p = make_piece(ptype, color)

    b[nr, nc] = p
    return b


def _update_castling(castling_rights, board, r, c, nr, nc):
    cr = castling_rights.copy()
    # King moved
    if board[r, c] in (W_KING, B_KING):
        side = "w" if board[r, c] == W_KING else "b"
        cr[side + "K"] = False
        cr[side + "Q"] = False
    # Rook moved or captured
    corners = {(7, 0): "wQ", (7, 7): "wK", (0, 0): "bQ", (0, 7): "bK"}
    for sq, key in corners.items():
        if (r, c) == sq or (nr, nc) == sq:
            cr[key] = False
    return cr


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class ChessGame:
    def __init__(self):
        self.board = initial_board()
        self.current_player = WHITE  # white starts
        self.move_history: List[str] = []
        self.board_states: List[np.ndarray] = [self.board.copy()]
        self.en_passant_sq: Optional[Tuple[int, int]] = None
        self.castling_rights = {"wK": True, "wQ": True, "bK": True, "bQ": True}
        self.halfmove_clock = 0  # for 50-move rule
        self.repetition: Dict[str, int] = {}

    def _board_key(self):
        return self.board.tobytes()

    def get_legal_moves(self) -> List[str]:
        color = self.current_player
        legal = []
        for r in range(8):
            for c in range(8):
                if piece_color(self.board[r, c]) != color:
                    continue
                for mv in _pseudo_legal_moves(
                    self.board, r, c,
                    self.en_passant_sq, self.castling_rights
                ):
                    nr, nc, special = mv
                    # For castling, also check that king doesn't pass through check
                    if special == "castle":
                        # Check king current square and intermediate square
                        back_row = 7 if color == WHITE else 0
                        mid_c = 5 if nc == 6 else 3
                        if _is_in_check(self.board, color):
                            continue
                        mid_board = _apply_move(
                            self.board, r, c, back_row, mid_c, "", color
                        )
                        if _is_in_check(mid_board, color):
                            continue
                    new_board = _apply_move(
                        self.board, r, c, nr, nc, special, color
                    )
                    if not _is_in_check(new_board, color):
                        # Build UCI string
                        uci = sq_to_uci(r, c) + sq_to_uci(nr, nc)
                        if special in ("q", "r", "b", "n"):
                            uci += special
                        legal.append(uci)
        return legal

    def play_move(self, uci: str) -> bool:
        r, c = uci_to_sq(uci[:2])
        nr, nc = uci_to_sq(uci[2:4])
        promo = uci[4] if len(uci) == 5 else ""

        p = self.board[r, c]
        ptype = piece_type(p)

        # Determine special type
        special = promo  # "" or "q"/"r"/"b"/"n"
        if ptype == 6 and abs(nc - c) == 2:
            special = "castle"
        elif ptype == 1 and nc != c and self.board[nr, nc] == EMPTY:
            special = "ep"

        new_board = _apply_move(
            self.board, r, c, nr, nc, special, self.current_player
        )

        # Update castling rights
        self.castling_rights = _update_castling(
            self.castling_rights, self.board, r, c, nr, nc
        )

        # Update en passant
        if ptype == 1 and abs(nr - r) == 2:
            mid_row = (r + nr) // 2
            self.en_passant_sq = (mid_row, c)
        else:
            self.en_passant_sq = None

        self.board = new_board
        self.move_history.append(uci)
        self.board_states.append(self.board.copy())
        self.current_player *= -1

        # 50-move rule tracking
        if ptype == 1 or self.board[nr, nc] != EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Repetition tracking
        key = self._board_key()
        self.repetition[key] = self.repetition.get(key, 0) + 1
        return True

    def is_game_over(self) -> bool:
        if not self.get_legal_moves():
            return True
        if self.halfmove_clock >= 100:  # 50-move rule
            return True
        if any(v >= 3 for v in self.repetition.values()):
            return True
        return False


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_random_game(
    rng: np.random.Generator, max_moves: int = 80
) -> Tuple[List[str], List[np.ndarray]]:
    """Play a random legal chess game. Returns (uci_moves, boards_after_each_move)."""
    game = ChessGame()
    while not game.is_game_over() and len(game.move_history) < max_moves:
        legal = game.get_legal_moves()
        if not legal:
            break
        move = legal[rng.integers(len(legal))]
        game.play_move(move)
    return game.move_history, game.board_states[1:]


def generate_dataset(
    n_games: int, seed: int = 42
) -> Tuple[List[List[str]], List[List[np.ndarray]]]:
    """Generate n_games random chess games."""
    rng = np.random.default_rng(seed)
    all_moves, all_boards = [], []
    for i in range(n_games):
        moves, boards = generate_random_game(rng)
        if len(moves) > 5:  # discard trivially short games
            all_moves.append(moves)
            all_boards.append(boards)
        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{n_games} games...")
    return all_moves, all_boards


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(all_moves: List[List[str]]) -> Dict[str, int]:
    """Build move -> token_id vocabulary from all games."""
    unique = sorted({m for game in all_moves for m in game})
    return {m: i for i, m in enumerate(unique)}


def save_vocab(vocab: Dict[str, int], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Generating 5 random chess games...")
    for i in range(5):
        moves, boards = generate_random_game(rng)
        print(f"Game {i}: {len(moves)} moves — last position:")
        for row in boards[-1]:
            print(" ".join(PIECE_SYMBOLS[int(p)] for p in row))
        print()
