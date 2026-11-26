import os
import json
import re
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from joblib import load
from sudoku import generate_full, create_puzzle, get_hint, solve, board_to_features, SIZE, compute_candidates

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

rf_model = None
xgb_model = None

def load_models():
    global rf_model, xgb_model
    rf_path = os.path.join("models", "random_forest.pkl")
    xgb_path = os.path.join("models", "xgboost.pkl")
    if os.path.exists(rf_path):
        rf_model = load(rf_path)
    if os.path.exists(xgb_path):
        xgb_model = load(xgb_path)

def decode_target(t):
    r = t // (SIZE * SIZE)
    rem = t % (SIZE * SIZE)
    c = rem // SIZE
    v = (rem % SIZE) + 1
    return int(r), int(c), int(v)

def ensemble_hint(board):
    feats = board_to_features(board).reshape(1, -1)
    votes = []
    if rf_model is not None:
        p = rf_model.predict(feats)[0]
        votes.append(int(p))
    if xgb_model is not None:
        if hasattr(xgb_model, "predict_proba"):
            probs = xgb_model.predict_proba(feats)[0]
            t = int(np.argmax(probs))
            votes.append(t)
        else:
            t = int(xgb_model.predict(feats)[0])
            votes.append(t)
    if votes:
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.items(), key=lambda x: x[1])[0]
        r, c, val = decode_target(best)
        if board[r, c] == 0:
            return r, c, val
    h = get_hint(board.copy())
    return h

def format_solution(board):
    lines = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            row.append(str(int(board[r, c])))
        lines.append(" ".join(row))
    return "\n".join(lines)

ORDINALS = {
    "first": 1, "1st": 1, "one": 1,
    "second": 2, "2nd": 2, "two": 2,
    "third": 3, "3rd": 3, "three": 3,
    "fourth": 4, "4th": 4, "four": 4,
    "fifth": 5, "5th": 5, "five": 5,
    "sixth": 6, "6th": 6, "six": 6,
    "seventh": 7, "7th": 7, "seven": 7,
    "eighth": 8, "8th": 8, "eight": 8,
    "ninth": 9, "9th": 9, "nine": 9
}

def extract_index(text, keys):
    for k in keys:
        m = re.search(rf"{k}\s*(\d)", text)
        if m:
            return int(m.group(1))
    for word, val in ORDINALS.items():
        for k in keys:
            if f"{word} {k}" in text or f"{k} {word}" in text:
                return val
    return None

def parse_position(text):
    row = extract_index(text, ["row", "r"])
    col = extract_index(text, ["col", "column", "c", "line"])
    return row, col

def answer_for_position(board, row, col):
    r = row - 1
    c = col - 1
    if r < 0 or r >= SIZE or c < 0 or c >= SIZE:
        return None
    if board[r, c] != 0:
        return f"Row {row}, col {col} is already {int(board[r, c])}."
    cands = compute_candidates(board)
    vals = cands.get((r, c), [])
    if len(vals) == 1:
        return f"Row {row}, col {col} should be {vals[0]}."
    if len(vals) > 1:
        return f"Row {row}, col {col} can be: {', '.join(map(str, vals))}."
    return "That cell has no valid values. The board may have conflicts."

def answer_for_row(board, row):
    r = row - 1
    if r < 0 or r >= SIZE:
        return None
    cands = compute_candidates(board)
    items = []
    for c in range(SIZE):
        if board[r, c] == 0:
            vals = cands.get((r, c), [])
            if vals:
                items.append(f"c{c+1}:{'/'.join(map(str, vals))}")
    if not items:
        return f"Row {row} is complete."
    return f"Row {row} empty cells candidates: " + ", ".join(items)

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/new-game", methods=["GET"])
def new_game():
    difficulty = request.args.get("difficulty", "medium")
    holes = 45
    if difficulty == "easy":
        holes = 35
    elif difficulty == "hard":
        holes = 55
    solved = generate_full()
    puzzle = create_puzzle(solved, holes=holes)
    return jsonify({
        "puzzle": puzzle.tolist(),
        "solved": solved.tolist()
    })

@app.route("/api/hint", methods=["POST"])
def api_hint():
    data = request.get_json(force=True)
    board = np.array(data.get("board"), dtype=int)
    h = ensemble_hint(board)
    if h is None:
        return jsonify({"hint": None})
    r, c, v = h
    return jsonify({"hint": {"row": r, "col": c, "val": v}})

@app.route("/api/validate-move", methods=["POST"])
def validate_move():
    data = request.get_json(force=True)
    board = np.array(data.get("board"), dtype=int)
    row = int(data.get("row"))
    col = int(data.get("col"))
    val = int(data.get("val"))
    if board[row, col] != 0:
        return jsonify({"valid": False, "reason": "cell filled"})
    if val < 1 or val > 9:
        return jsonify({"valid": False, "reason": "value range"})
    ok = True
    if val in board[row, :]:
        ok = False
    if val in board[:, col]:
        ok = False
    br = (row // 3) * 3
    bc = (col // 3) * 3
    if val in board[br:br+3, bc:bc+3]:
        ok = False
    return jsonify({"valid": ok})

@app.route("/api/solve", methods=["POST"])
def api_solve():
    data = request.get_json(force=True)
    board = np.array(data.get("board"), dtype=int)
    b = board.copy()
    if solve(b):
        return jsonify({"solution": b.tolist()})
    return jsonify({"solution": None})

def chatbot_response(message, board):
    m = (message or "").lower()
    row, col = parse_position(m)
    if row and col:
        res = answer_for_position(board, row, col)
        if res:
            return res
    if row and not col:
        res = answer_for_row(board, row)
        if res:
            return res
    if any(k in m for k in ["hint", "help", "which number", "what number", "input", "fill", "give me"]):
        h = ensemble_hint(board.copy())
        if h is None:
            return "No hint available. Try another strategy."
        r, c, v = h
        return f"Try row {r+1}, col {c+1} with {v}."
    if "strategy" in m or "how" in m:
        return "Fill singles first, then use row/column elimination and box scanning."
    if "solve" in m or "solution" in m:
        b = board.copy()
        if solve(b):
            grid = format_solution(b)
            return "Here is the solved board:\n" + grid
        return "Current board may be invalid. Check conflicts."
    return "Ask for a hint, strategy tips, or solve."

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    board = np.array(data.get("board"), dtype=int)
    reply = chatbot_response(message, board)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=8000)
