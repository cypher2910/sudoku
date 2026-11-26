import random
import numpy as np

SIZE = 9
BOX = 3

def valid(board, r, c, v):
    if v in board[r, :]:
        return False
    if v in board[:, c]:
        return False
    br = (r // BOX) * BOX
    bc = (c // BOX) * BOX
    if v in board[br:br+BOX, bc:bc+BOX]:
        return False
    return True

def find_empty(board):
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r, c] == 0:
                return r, c
    return None

def solve(board):
    pos = find_empty(board)
    if not pos:
        return True
    r, c = pos
    nums = list(range(1, SIZE+1))
    random.shuffle(nums)
    for v in nums:
        if valid(board, r, c, v):
            board[r, c] = v
            if solve(board):
                return True
            board[r, c] = 0
    return False

def solution_count(board):
    count = 0
    def backtrack(b):
        nonlocal count
        if count > 1:
            return
        pos = find_empty(b)
        if not pos:
            count += 1
            return
        r, c = pos
        for v in range(1, SIZE+1):
            if valid(b, r, c, v):
                b[r, c] = v
                backtrack(b)
                b[r, c] = 0
    backtrack(board.copy())
    return count

def generate_full():
    board = np.zeros((SIZE, SIZE), dtype=int)
    for i in range(SIZE):
        nums = list(range(1, SIZE+1))
        random.shuffle(nums)
        for j in range(SIZE):
            board[i, j] = 0
    solve(board)
    return board

def create_puzzle(solved, holes=40):
    puzzle = solved.copy()
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    random.shuffle(cells)
    removed = 0
    for r, c in cells:
        if removed >= holes:
            break
        temp = puzzle[r, c]
        puzzle[r, c] = 0
        if solution_count(puzzle.copy()) != 1:
            puzzle[r, c] = temp
        else:
            removed += 1
    return puzzle

def compute_candidates(board):
    cands = {}
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r, c] == 0:
                vals = []
                for v in range(1, SIZE+1):
                    if valid(board, r, c, v):
                        vals.append(v)
                cands[(r, c)] = vals
    return cands

def get_hint(board):
    cands = compute_candidates(board)
    if not cands:
        return None
    single = [(pos, vals[0]) for pos, vals in cands.items() if len(vals) == 1]
    if single:
        (r, c), v = single[0]
        return r, c, v
    solved = board.copy()
    if solve(solved):
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r, c] == 0:
                    return r, c, solved[r, c]
    return None

def board_to_features(board):
    flat = board.flatten()
    zeros = np.count_nonzero(flat == 0)
    row_zeros = [np.count_nonzero(board[i, :] == 0) for i in range(SIZE)]
    col_zeros = [np.count_nonzero(board[:, j] == 0) for j in range(SIZE)]
    box_zeros = []
    for br in range(0, SIZE, BOX):
        for bc in range(0, SIZE, BOX):
            box_zeros.append(np.count_nonzero(board[br:br+BOX, bc:bc+BOX] == 0))
    feats = np.concatenate([flat, [zeros], row_zeros, col_zeros, box_zeros])
    return feats.astype(np.int16)

def generate_dataset(n=1000, holes_range=(40, 55)):
    X = []
    y = []
    for _ in range(n):
        solved = generate_full()
        holes = random.randint(holes_range[0], holes_range[1])
        puzzle = create_puzzle(solved, holes=holes)
        hint = get_hint(puzzle.copy())
        if hint is None:
            continue
        r, c, v = hint
        feats = board_to_features(puzzle)
        target = r * SIZE * SIZE + c * SIZE + (v - 1)
        X.append(feats)
        y.append(target)
    return np.array(X), np.array(y)
