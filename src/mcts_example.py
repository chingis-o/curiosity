import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.available_moves())

    def best_child(self, exploration_param=1.4):
        choices_weights = [(c.wins / c.visits) + exploration_param * math.sqrt((2 * math.log(self.visits) / c.visits)) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

def mcts(root, iterations):
    for _ in range(iterations):
        node = root
        state = root.state.copy()

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            state.make_move(node.state, 'X' if state.current_winner == 'O' else 'O')

        # Expansion
        if not state.empty_squares():
            continue

        possible_moves = state.available_moves()
        move = node.rollout_policy(possible_moves)
        child_state = state.copy()
        child_state.make_move(move, 'X' if state.current_winner == 'O' else 'O')
        child_node = node.add_child(child_state)

        # Simulation
        result = simulate_random_playout(child_state)

        # Backpropagation
        backpropagate(child_node, result)

def simulate_random_playout(state):
    current_state = state.copy()
    while current_state.empty_squares():
        possible_moves = current_state.available_moves()
        move = random.choice(possible_moves)
        current_state.make_move(move, 'X' if current_state.current_winner == 'O' else 'O')
    if current_state.current_winner == 'X':
        return 1
    elif current_state.current_winner == 'O':
        return -1
    else:
        return 0

def backpropagate(node, result):
    while node is not None:
        node.update(result)
        node = node.parent

# Example usage
if __name__ == "__main__":
    game = TicTacToe()
    root = Node(game)
    mcts(root, iterations=1000)
    best_move = root.best_child().state
    print("Best move:", best_move)