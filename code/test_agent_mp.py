"""
Test that dynamically loaded agents work correctly with multiprocessing.
This mirrors how main.py actually uses them.
"""
import multiprocessing
import queue
import sys
import os

# Fix Windows console encoding for printing unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# This must be at module level for multiprocessing spawn to find it
def agent_process_wrapper(agent_fn, board_copy, player_symbol, result_queue):
    try:
        move = agent_fn(board_copy, player_symbol)
        result_queue.put(move)
    except Exception as e:
        result_queue.put(e)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Headless pygame init
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.display.init()
    pygame.font.init()

    from agent_loader import load_agent
    from game import XOShiftGame

    print("=" * 50)
    print("Test 1: Loading sample_agent.py via multiprocessing")
    print("=" * 50)

    agent = load_agent("code/sample_agent.py")
    print(f"  Agent loaded: {agent}")

    game = XOShiftGame(3)
    board_copy = [[cell for cell in row] for row in game.board]
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=agent_process_wrapper,
        args=(agent, board_copy, "X", result_queue),
    )
    process.start()

    try:
        move = result_queue.get(timeout=5)
        process.join()
        print(f"  Move received: {move}")
        print("  ✅ Test 1 PASSED")
    except queue.Empty:
        process.terminate()
        process.join()
        print("  ❌ Test 1 FAILED: Queue empty (timeout)")
        sys.exit(1)
    except Exception as e:
        process.terminate()
        process.join()
        print(f"  ❌ Test 1 FAILED: {e}")
        sys.exit(1)

    print()
    print("=" * 50)
    print("Test 2: Loading your_agent.py (advanced AI) via multiprocessing")
    print("=" * 50)

    agent2 = load_agent("code/your_agent.py")
    print(f"  Agent loaded: {agent2}")

    game2 = XOShiftGame(3)
    board_copy2 = [[cell for cell in row] for row in game2.board]
    result_queue2 = multiprocessing.Queue()
    process2 = multiprocessing.Process(
        target=agent_process_wrapper,
        args=(agent2, board_copy2, "X", result_queue2),
    )
    process2.start()

    try:
        move2 = result_queue2.get(timeout=10)
        process2.join()
        print(f"  Move received: {move2}")
        print("  ✅ Test 2 PASSED")
    except queue.Empty:
        process2.terminate()
        process2.join()
        print("  ❌ Test 2 FAILED: Queue empty (timeout)")
        sys.exit(1)
    except Exception as e:
        process2.terminate()
        process2.join()
        print(f"  ❌ Test 2 FAILED: {e}")
        sys.exit(1)

    print()
    print("=" * 50)
    print("Test 3: Both agents in an agent-agent game simulation")
    print("=" * 50)

    agent_x = load_agent("code/sample_agent.py")
    agent_o = load_agent("code/your_agent.py")
    game3 = XOShiftGame(3)
    players = ["X", "O"]
    agents = {"X": agent_x, "O": agent_o}
    current = 0

    for turn in range(6):  # 3 turns each
        symbol = players[current]
        board_copy3 = [[cell for cell in row] for row in game3.board]
        rq = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=agent_process_wrapper,
            args=(agents[symbol], board_copy3, symbol, rq),
        )
        p.start()

        try:
            mv = rq.get(timeout=10)
            p.join()
            game3.apply_move(*mv, symbol)
            if not game3.winner:
                game3.switch_player()
            current = game3.current_player_index
            print(f"  Turn {turn + 1}: {symbol} {mv} -> winner={game3.winner}")
        except queue.Empty:
            p.terminate()
            p.join()
            print(f"  ❌ Turn {turn + 1} FAILED: timeout")
            sys.exit(1)

    print("  ✅ Test 3 PASSED (simulated game ran without errors)")

    print()
    print("=" * 50)
    print("Test 4: BOTH agents using the SAME file (the crash scenario)")
    print("=" * 50)

    # This is exactly what happens when agent1 and agent2 both point to sample_agent.py
    agent_same_1 = load_agent("code/sample_agent.py")
    agent_same_2 = load_agent("code/sample_agent.py")
    print(f"  agent1: {agent_same_1}")
    print(f"  agent2: {agent_same_2}")
    print(f"  Same object? {agent_same_1 is agent_same_2}")

    if agent_same_1 is not agent_same_2:
        print("  ❌ Test 4 FAILED: Agents should be the same object")
        sys.exit(1)

    # Run both in separate subprocesses simultaneously
    game4 = XOShiftGame(3)
    rq_a = multiprocessing.Queue()
    rq_b = multiprocessing.Queue()
    pa = multiprocessing.Process(
        target=agent_process_wrapper,
        args=(agent_same_1, [[c for c in row] for row in game4.board], "X", rq_a),
    )
    pb = multiprocessing.Process(
        target=agent_process_wrapper,
        args=(agent_same_2, [[c for c in row] for row in game4.board], "O", rq_b),
    )
    pa.start()
    pb.start()

    try:
        ma = rq_a.get(timeout=5)
        mb = rq_b.get(timeout=5)
        pa.join()
        pb.join()
        print(f"  agent1 move: {ma}")
        print(f"  agent2 move: {mb}")
        print("  ✅ Test 4 PASSED")
    except queue.Empty as e:
        pa.terminate()
        pb.terminate()
        pa.join()
        pb.join()
        print(f"  ❌ Test 4 FAILED: Queue empty - {e}")
        sys.exit(1)

    print()
    print("=" * 50)
    print("🎉 ALL 4 TESTS PASSED!")
    print("=" * 50)

    pygame.quit()
