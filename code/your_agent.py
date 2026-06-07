import random
import time
from typing import List, Optional, Tuple, Dict
from agent_utils import get_all_valid_moves

MAX_DEPTH = 4
AGENT_TIME_LIMIT = 2.0
TIME_MARGIN = 0.5

Move = Tuple[int, int, int, int]
Board = List[List[Optional[str]]]
Symbol = str

random.seed(0)


def agent_move(board: Board, player_symbol: Symbol) -> Move:
    start_time = time.time()
    size = len(board)
    Z = [[[random.getrandbits(64) for _ in range(2)] for _ in range(size)] for _ in range(size)]

    LINES: List[List[Tuple[int, int]]] = []
    for i in range(size):
        LINES.append([(i, j) for j in range(size)])
        LINES.append([(j, i) for j in range(size)])
    LINES.append([(i, i) for i in range(size)])
    LINES.append([(i, size - 1 - i) for i in range(size)])

    weights: List[List[int]] = [[1] * size for _ in range(size)]
    for r in (0, size - 1):
        for c in (0, size - 1):
            weights[r][c] = size // 2
    if size % 2:
        weights[size // 2][size // 2] = size

    def check_winner(bd: Board) -> Optional[Symbol]:
        for line in LINES:
            vals = [bd[rr][cc] for rr, cc in line]
            if vals[0] is not None and all(v == vals[0] for v in vals):
                return vals[0]
        return None

    def my_hash(bd: Board) -> int:
        h = 0
        for rr in range(size):
            for cc in range(size):
                p = bd[rr][cc]
                if p:
                    idx = 0 if p == 'X' else 1
                    h ^= Z[rr][cc][idx]
        return h

    def time_up() -> bool:
        return (time.time() - start_time) > (AGENT_TIME_LIMIT - TIME_MARGIN)

    def other_player(player: Symbol) -> Symbol:
        return 'O' if player == 'X' else 'X'

    def simulate(bd: Board, move: Move, pl: Symbol) -> Board:
        sr, sc, tr, tc = move
        new_bd = [row.copy() for row in bd]
        if sr == tr and sc == tc:
            return new_bd
        if sr == tr:
            step = 1 if sc < tc else -1
            for cc in range(sc, tc, step):
                new_bd[sr][cc] = new_bd[sr][cc + step]
        else:
            step = 1 if sr < tr else -1
            for rr in range(sr, tr, step):
                new_bd[rr][sc] = new_bd[rr + step][sc]
        new_bd[tr][tc] = pl
        return new_bd

    def evaluate(bd: Board, my_moves: Optional[List[Move]] = None,
                 opp_moves: Optional[List[Move]] = None) -> float:
        opp = other_player(player_symbol)
        pc = sum(row.count(player_symbol) - row.count(opp) for row in bd)
        if my_moves is None:
            my_moves = get_all_valid_moves(bd, player_symbol)
        if opp_moves is None:
            opp_moves = get_all_valid_moves(bd, opp)
        mob = len(my_moves) - len(opp_moves)

        def count_almost(sym: Symbol) -> int:
            cnt = 0
            for line in LINES:
                vals = [bd[rrr][ccc] for rrr, ccc in line]
                if vals.count(sym) == size - 1 and vals.count(None) == 1:
                    cnt += 1
            return cnt

        thr = count_almost(player_symbol) - count_almost(opp)
        pos = 0
        for rr in range(size):
            for cc in range(size):
                if bd[rr][cc] == player_symbol:
                    pos += weights[rr][cc]
                elif bd[rr][cc] == opp:
                    pos -= weights[rr][cc]
        return 1.0 * pc + 0.8 * mob + 1.2 * thr + 0.5 * pos

    table: Dict[Tuple[int, int, Symbol], Tuple[int, float, Optional[Move]]] = {}

    def order_moves(bd: Board, moves: List[Move], pl: Symbol) -> List[Move]:
        win_moves, loose_moves, rest = [], [], []
        opp = other_player(pl)
        for move in moves:
            child = simulate(bd, move, pl)
            if check_winner(child) == pl:
                win_moves.append(move)
            else:
                opp_moves = get_all_valid_moves(child, opp)
                if any(check_winner(simulate(child, om, opp)) == opp for om in opp_moves):
                    loose_moves.append(move)
                else:
                    rest.append(move)
        rest.sort(key=lambda m: evaluate(simulate(bd, m, pl)), reverse=pl == 'X')
        return win_moves + rest + loose_moves

    def minimax(bd: Board, depth: int, maximizing: bool, pl: Symbol,
                  alpha: float, beta: float, hash_val: int) -> Tuple[float, Optional[Move], bool]:
        key = (hash_val, depth, pl)
        if key in table:
            d_stored, v_stored, m_stored = table[key]
            if d_stored >= depth:
                return v_stored, m_stored, True
        if time_up():
            v = evaluate(bd)
            return v, None, False
        winner = check_winner(bd)
        if winner == player_symbol:
            return float('inf'), None, True
        if winner == other_player(player_symbol):
            return float('-inf'), None, True
        if depth == 0:
            v = evaluate(bd)
            return v, None, True
        moves = get_all_valid_moves(bd, pl)
        if not moves:
            v = evaluate(bd, my_moves=[], opp_moves=get_all_valid_moves(bd, other_player(pl)))
            return v, None, True
        ordered = order_moves(bd, moves, pl)
        best_mv: Optional[Move] = ordered[0]
        if maximizing:
            value = float('-inf')
            for move in ordered:
                if time_up():
                    return value, best_mv, False
                child = simulate(bd, move, pl)
                h2 = my_hash(child)
                res, _, complete = minimax(child, depth - 1, False, other_player(pl),
                                            alpha, beta, h2)
                if not complete:
                    return value, best_mv, False
                if res > value:
                    value, best_mv = res, move
                    alpha = max(alpha, value)
                if alpha >= beta:
                    break
            table[(hash_val, depth, pl)] = (depth, value, best_mv)
            return value, best_mv, True
        else:
            value = float('inf')
            for move in ordered:
                if time_up():
                    return value, best_mv, False
                child = simulate(bd, move, pl)
                h2 = my_hash(child)
                res, _, complete = minimax(child, depth - 1, True, other_player(pl),
                                            alpha, beta, h2)
                if not complete:
                    return value, best_mv, False
                if res < value:
                    value, best_mv = res, move
                    beta = min(beta, value)
                if alpha >= beta:
                    break
            table[(hash_val, depth, pl)] = (depth, value, best_mv)
            return value, best_mv, True

    valid_moves = get_all_valid_moves(board, player_symbol)
    if not valid_moves:
        return 0, 0, 0, 0
    best_overall: Move = valid_moves[0]
    initial_hash = my_hash(board)
    for d in range(1, MAX_DEPTH + 1):
        if time_up():
            break
        val, mv, completed = minimax(board, d, True, player_symbol, float('-inf'), float('inf'), initial_hash)
        if completed and mv:
            best_overall = mv
        else:
            break
    return best_overall
