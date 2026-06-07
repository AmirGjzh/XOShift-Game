# XOShift Game ♟️

**XOShift** is an adversarial two-player board game with a unique "shift" mechanic — instead of simply placing pieces, players **push pieces along rows and columns** from one edge of the board to another. This project features a fully playable Pygame GUI, support for human and AI players, a replay system, and a competitive AI agent powered by **Minimax search with Alpha-Beta pruning, iterative deepening, and Zobrist hashing**.

---

## 🎮 Game Rules

### Board
- Played on an **N×N** grid (N = 3, 4, or 5).
- The **rim** (outermost ring of cells) is the only interactive area.

### Moves
A move consists of two steps:

1. **Select a source cell** on the rim:
   - If any rim cell is **empty**, you **must** select an empty one.
   - If all rim cells are occupied, you must select **one of your own pieces** on the rim.

2. **Push to a target cell** on the rim:
   - You shift all pieces in the selected row/column one step toward the target, then place your symbol at the target.

> **Example:** Selecting cell `(0, 2)` and pushing left to `(0, 0)` shifts the pieces in row 0 left by one, and your piece ends up at `(0, 0)`.

### Win Condition
- **N-in-a-row:** Fill an entire row, column, or diagonal with your symbol (`X` or `O`).
- Maximum **250 turns** before the game is declared a draw.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Three Board Sizes** | 3×3, 4×4, and 5×5 |
| **Three Game Modes** | Human vs Human, Human vs Agent, Agent vs Agent |
| **Replay System** | Record games (JSON) and step through them with ⬅️/➡️ arrows |
| **Custom AI Agents** | Drop-in Python agents — dynamically loaded at runtime |
| **Sample Random Agent** | Baseline agent that picks random valid moves |
| **Competitive AI Agent** | Minimax + Alpha-Beta + Zobrist hashing + iterative deepening |
| **Pygame GUI** | Interactive board with hover highlights, selection states, and game-over overlay |

---

## 🧠 AI Agent: `your_agent.py`

The intelligent agent (`your_agent.py`) combines several advanced game-AI techniques:

### 1. Minimax Algorithm
A recursive depth-first search that explores the game tree. The agent assumes the opponent plays optimally and chooses moves that maximize its own minimum guaranteed outcome.

### 2. Alpha-Beta Pruning
Eliminates branches that cannot possibly influence the final decision, dramatically reducing the number of nodes evaluated. The agent uses two bounds:
- **α (alpha):** Best score the maximizing player can guarantee
- **β (beta):** Best score the minimizing player can guarantee

### 3. Iterative Deepening (IDDFS)
The agent searches to increasing depths (1 → 2 → 3 → 4) within the **2-second time limit**. If time runs out, it falls back to the best move found at the previous completed depth.

### 4. Move Ordering
Moves are ordered to maximize pruning efficiency:
- **Winning moves** (immediate win) come first
- **Rest** sorted by evaluation score
- **Losing moves** (opponent can win next turn) come last

### 5. Evaluation Function
A weighted heuristic that evaluates non-terminal board states:

| Component | Weight | Description |
|---|---|---|
| **Piece Count** | ×1.0 | Difference in number of pieces on the board |
| **Mobility** | ×0.8 | Difference in number of legal moves available |
| **Threats** | ×1.2 | Difference in "almost-winning" lines (N−1 of a kind) |
| **Position** | ×0.5 | Positional value: corners and center weighted higher |

### 6. Zobrist Hashing + Transposition Table
Each board state is hashed using **Zobrist hashing** (random 64-bit values per cell/symbol). A transposition table caches evaluated positions to avoid redundant computation across search branches.

### 7. Time Management
The agent runs in a **separate process** with a strict 2-second timeout. If the agent exceeds the limit, its turn is skipped.

---

## 🔧 Project Structure

```
XOShift-Game/
├── code/
│   ├── main.py                    # Game loop, event handling, agent orchestration
│   ├── game.py                    # XOShiftGame (board rules, move logic, win detection)
│   ├── ui.py                      # Pygame UI (menu, board rendering, replay browser)
│   ├── utils.py                   # Font loading, text rendering helpers
│   ├── agent_loader.py            # Dynamic agent module loader
│   ├── agent_utils.py             # Agent helpers (valid move enumeration)
│   ├── sample_agent.py            # Random baseline agent
│   ├── your_agent.py              # Competitive AI agent (Minimax + Alpha-Beta)
│   └── test_agent_mp.py           # Multiprocessing agent tests
├── assets/
│   └── Alegreya-Regular.otf   # Custom display font
├── replays/                   # Saved game replays (JSON) → created on first run
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── Report.pdf                 # Project report
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.8+** (tested with Python 3.10+)
- **pip** (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/XOShift-Game.git
cd XOShift-Game

# 2. (Recommended) Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the game
python code/main.py
```

### Dependencies
- **pygame-ce** (≥2.5.0) — Community Edition fork of Pygame. Used for all graphics, input handling, and rendering.

> **Note:** If you're using Python 3.14, `pygame` (original) has no pre-built wheels yet, so `pygame-ce` is required. On Python ≤3.13, you can use either `pygame` or `pygame-ce`.

---

## 🕹️ How to Play

1. **Main Menu** — Choose:
   - Board size (3×3, 4×4, or 5×5)
   - Game mode (Human vs Human, Human vs Agent, Agent vs Agent)
   - Toggle replay recording

2. **Playing** — Click a **rim cell** to select it, then click a **target rim cell** to push the piece.

3. **Agent Matches** — Agent moves are computed automatically with a 2-second thinking limit.

4. **Game Over** — The winner is announced with an overlay. Press **Return** or click the button to return to the menu.

5. **Replays** — Navigate through recorded games using the ⬅️ / ➡️ arrow keys in replay mode.

---

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

Then select your agent in the **Agent vs Agent** or **Human vs Agent** mode — the game automatically loads files via `agent_loader.py`. Edit `main.py` to point to your agent file:

```python
agent1_path_config = "code/your_agent.py"
agent2_path_config = "code/sample_agent.py"
```

---

## 📊 Replay System

- Replays are saved as JSON files in the `replays/` directory.
- Format:
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
      {"player": "O", "src_r": 4, "src_c": 1, "tgt_r": 4, "tgt_c": 0},
      ...
    ]
  }
  ```
- Browse and step through replays from the **Replay a Game** menu option.

---

## 🧪 Testing

The project was tested across all board sizes (3×3, 4×4, 5×5) and game modes:
- **Human vs Human** — Full manual play verification
- **Human vs Agent** — Agent within time limit, valid moves only
- **Agent vs Agent** — Two agents playing autonomously with replay recording
- **Replay System** — Forward/backward stepping, load/restart flow
- **Edge Cases** — Draw after 250 turns, board full, timeout handling

---

## 📝 License

This project was developed as an AI course project. Feel free to use, modify, and extend.

---

## 👤 Author

**Amir Mohammad Ganjizade** — AI Project 402243093
