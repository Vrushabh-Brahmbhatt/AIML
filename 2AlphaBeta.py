MAX, MIN = 1000, -1000

def minmax(depth, nodeIndex, maximizing_player, values, alpha, beta, max_depth):
    # Base case: when depth equals max depth, return the value of the node
    if depth == max_depth:
        return values[nodeIndex]
    
    # Maximizing player
    if maximizing_player:
        best = MIN
        for i in range(0, 2):  # Two children for each node
            val = minmax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta, max_depth)
            best = max(best, val)
            alpha = max(alpha, best)
            
            # Beta Cutoff
            if beta <= alpha:
                break
        return best
    else:  # Minimizing player
        best = MAX
        for i in range(0, 2):  # Two children for each node
            val = minmax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta, max_depth)
            best = min(best, val)
            beta = min(beta, best)
            
            # Alpha Cutoff
            if beta <= alpha:
                break
        return best

if __name__ == '__main__':
    # Taking input from user
    max_depth = int(input("Enter the maximum depth of the tree (e.g., 3): "))
    
    num_nodes = 2 ** max_depth  # Number of leaf nodes at the last level
    print(f"Enter {num_nodes} values for the leaf nodes:")

    values = []
    for i in range(num_nodes):
        val = int(input(f"Value for leaf node {i+1}: "))
        values.append(val)
    
    # Run the Minimax algorithm with Alpha-Beta pruning
    print("The optimal value is ", minmax(0, 0, True, values, MIN, MAX, max_depth))

