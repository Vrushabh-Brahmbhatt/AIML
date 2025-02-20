import numpy as np
from queue import PriorityQueue

def get_moves(state):
    zero_pos = np.argwhere(state == 0)[0]
    moves = []
    for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        new_pos = zero_pos + d
        if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:
            new_state = state.copy()
            new_state[zero_pos[0], zero_pos[1]], new_state[new_pos[0], new_pos[1]] = new_state[new_pos[0], new_pos[1]], new_state[zero_pos[0], zero_pos[1]]
            moves.append(tuple(map(tuple, new_state)))
    return moves

def heuristic(state, goal_state):
    return sum(1 for i in range(3) for j in range(3) if state[i][j] != goal_state[i][j])

def solve_puzzle(initial_state, goal_state):
    queue = PriorityQueue()
    init = tuple(map(tuple, initial_state))
    goal_t = tuple(map(tuple, goal_state))
    queue.put((0, init, []))
    visited = set()

    while not queue.empty():
        cost, state_tuple, path = queue.get()
        if state_tuple == goal_t:
            return path + [state_tuple]
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for move in get_moves(np.array(state_tuple)):
            new_cost = len(path) + 1 + heuristic(move, goal_state)
            queue.put((new_cost, move, path + [state_tuple]))
    return None

# Hardcoded initial and goal states for the 8-puzzle
initial_state = np.array([[2, 8, 1],
                          [0, 4, 3],
                          [7, 6, 5]])

goal_state = np.array([[1, 2, 3],
                       [8, 0, 4],
                       [7, 6, 5]])

solution = solve_puzzle(initial_state, goal_state)
if solution:
    for step in solution:
        print(np.array(step), "\n")
    print("Number of moves:", len(solution) - 1)
else:
    print("No solution found.")
