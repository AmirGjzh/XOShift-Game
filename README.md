# ♟️ XOShift

XOShift is a two-player board game with a twist: instead of just placing pieces, you push them along rows and columns from one edge of the board to the other. This repo has a full Pygame GUI, human and AI players, a replay system, and a competitive AI agent built on Minimax with Alpha-Beta pruning, iterative deepening, and Zobrist hashing.

---

## ✨ Features

| Feature | Description |
|---|---|
| Three board sizes | 3×3, 4×4, and 5×5 |
| Three game modes | Human vs Human, Human vs Agent, Agent vs Agent |
| Replay system | Games are saved as JSON and can be stepped through with ⬅️ / ➡️ |
| Custom agents | Drop in your own Python agent — loaded dynamically at runtime |
| Sample agent | A baseline agent that just picks random valid moves |
| Competitive agent | Minimax + Alpha-Beta + Zobrist hashing + iterative deepening |
| Pygame GUI | Hover highlights, selection states, game-over overlay |

## 📁 Project Structure

```
XOShift-Game/
├── code/
│   ├── main.py                    # Game loop, event handling, agent orchestration
│   ├── game.py                    # XOShiftGame — board rules, move logic, win detection
│   ├── ui.py                      # Pygame UI (menu, board rendering, replay browser)
│   ├── utils.py                   # Font loading, text rendering helpers
│   ├── agent_loader.py            # Dynamic agent module loader
│   ├── agent_utils.py             # Agent helpers (valid move enumeration)
│   ├── sample_agent.py            # Random baseline agent
│   ├── your_agent.py              # Competitive AI agent (Minimax + Alpha-Beta)
│   └── test_agent_mp.py           # Multiprocessing agent tests
├── assets/
│   └── Alegreya-Regular.otf       # Custom display font
├── replays/                       # Saved game replays (JSON), created on first run
├── requirements.txt
├── .gitignore
├── README.md
└── Report.pdf
```

## 🚀 Getting Started

**Prerequisites**
- Python 3.8+ (tested on 3.10+)
- pip

**Install**

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

python code/main.py
```

**Dependencies**

`pygame-ce` (≥2.5.0) — the Community Edition fork of Pygame, used for all graphics and input.

> On Python 3.14, regular `pygame` doesn't have prebuilt wheels yet, so `pygame-ce` is the one to use. On 3.13 and below, either works.

## 🎮 Game Rules

**Board** — an N×N grid (N = 3, 4, or 5). Only the rim (the outermost ring of cells) is interactive.

**A move has two steps:**

1. **Pick a source cell on the rim.**
   - If any rim cell is empty, you have to pick an empty one.
   - If the rim is full, you pick one of your own pieces instead.
2. **Push it to a target cell on the rim.**
   - Every piece in that row/column shifts one step toward the target, then your symbol lands on the target.

> Example: picking (0, 2) and pushing left to (0, 0) shifts row 0 left by one, and your piece ends up at (0, 0).

**Winning** — get N of your symbol in a row, column, or diagonal. If nobody wins within 250 turns, it's a draw.

## ▶️ How to Play

1. **Main menu** — pick your board size, game mode, and whether to record a replay.
2. **Playing** — click a rim cell to select it, then click a target rim cell to push your piece there.
3. **Agent matches** — agents move automatically, with a 2-second thinking limit each turn.
4. **Game over** — the winner shows up on an overlay. Press Return or click the button to head back to the menu.
5. **Replays** — step through a recorded game with the ⬅️ / ➡️ arrow keys.

## 🧠 How the AI Agent Works

`your_agent.py` is the competitive agent, and it combines a few classic game-AI techniques:

**1. Minimax** — explores the game tree assuming the opponent plays optimally, and picks the move that maximizes its own guaranteed outcome.

**2. Alpha-Beta pruning** — cuts off branches that can't affect the final decision, using two bounds: α (best score the maximizer can guarantee) and β (best score the minimizer can guarantee).

**3. Iterative deepening** — searches depth 1, then 2, then 3, and so on within a 2-second budget. If time runs out mid-search, it falls back to the best move found at the last completed depth.

**4. Move ordering** — moves are sorted to prune more effectively: immediate wins first, then by evaluation score, with moves that hand the opponent a win last.

**5. Evaluation function** — a weighted heuristic for non-terminal positions:

| Component | Weight | What it measures |
|---|---|---|
| Piece count | ×1.0 | Difference in pieces on the board |
| Mobility | ×0.8 | Difference in legal moves available |
| Threats | ×1.2 | Difference in "one away from winning" lines |
| Position | ×0.5 | Corners and center weighted higher |

**6. Zobrist hashing + transposition table** — each board state gets hashed (random 64-bit values per cell/symbol), and a transposition table caches evaluated positions so the search doesn't redo work across branches.

**7. Time management** — the agent runs in a separate process with a strict 2-second timeout. If it goes over, that turn is skipped.

## 🤖 Writing Your Own Agent

Create a Python file with an `agent_move` function:

```python
from typing import List, Optional, Tuple
from agent_utils import get_all_valid_moves

def agent_move(board: List[List[Optional[str]]], player_symbol: str) -> Tuple[int, int, int, int]:
    """
    Args:
        board:      Current board state (N×N list of 'X', 'O', or None)
        player_symbol: The agent's symbol ('X' or 'O')

    Returns:
        A tuple (src_row, src_col, tgt_row, tgt_col) representing the move.
    """
    valid_moves = get_all_valid_moves(board, player_symbol)
    # Choose a move...
    return valid_moves[0]
```

Then point `main.py` at your file — it gets loaded automatically via `agent_loader.py`:

```python
agent1_path_config = "code/your_agent.py"
agent2_path_config = "code/sample_agent.py"
```

## 📊 Replay System

Replays are saved as JSON in `replays/`:

```json
{
  "metadata": {
    "board_size": 5,
    "game_mode": "human-agent",
    "player_x_type": "human",
    "player_o_type": "your_agent",
    "winner": "X"
  },
  "moves": [
    {"player": "X", "src_r": 0, "src_c": 2, "tgt_r": 0, "tgt_c": 4},
    {"player": "O", "src_r": 4, "src_c": 1, "tgt_r": 4, "tgt_c": 0}
  ]
}
```

Browse and step through them from the **Replay a Game** menu option.

## 🧪 Testing

Tested across all board sizes and game modes:

- **Human vs Human** — full manual play verification
- **Human vs Agent** — agent stays within the time limit and only makes valid moves
- **Agent vs Agent** — two agents playing autonomously, with replay recording
- **Replay system** — forward/backward stepping, load/restart flow
- **Edge cases** — draw at 250 turns, full board, timeout handling

## 🤝 Contributing

Found a bug or have an idea? Open an issue or send a pull request — contributions are always welcome.

## 📜 License

This project is licensed under the MIT License.
