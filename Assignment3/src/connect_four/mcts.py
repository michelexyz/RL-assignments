import random
from typing import Tuple

import numpy as np
from connect_four.game_board import GameBoard
from connect_four.node import UCB, Greedy, Node, SelectionStrategy

game_init = np.array(
    [
        [2, 2, 2, 1, 0, 1, 0],
        [2, 1, 1, 1, 0, 2, 0],
        [1, 2, 2, 2, 0, 1, 0],
        [2, 1, 1, 1, 0, 2, 0],
        [1, 1, 1, 2, 0, 2, 0],
        [2, 2, 1, 2, 0, 1, 0],
    ]
)


class MCTS:

    root: Node
    game: GameBoard
    _is_trained: bool

    # def __init__(self) -> None:
    #     self.game = GameBoard.from_grid(game_init)
    #     self.root = Node.from_root(state=self.game.snapshot())
    
    def __init__(self, game_state=None) -> None:
        if game_state is None:
            game_state = game_init
        self.game = GameBoard.from_grid(game_state)
        self.root = Node.from_root(state=self.game.snapshot())
        

    def select(self, s: SelectionStrategy) -> Node:
        """Returns either a LEAF or TERMINAL node"""
        # Start always from the top
        node = self.root.select(strategy=s)
        # Iterate through nodes until reaching a leaf / terminal node
        while not (node.is_leaf or node.is_terminal):
            node = node.select(strategy=s)

        # VERY important: set game state to start playing from
        self.game.set_state(node.game_state)

        return node

    def expand(self, parent: Node) -> Node:
        """Expands LEAF node. Make sure TERMINAL nodes are not expanded
        A LEAF node always has available actions to take, by definition
        """
        # Cancel expanding if node is terminal
        if parent.is_terminal:
            return parent

        # Choose random action (from available) to expand parent and play one round
        action = random.choice(list(parent.available_actions))
        self.game.play(action=action)

        # Create the child with the results from the just played game
        child = Node.from_parent(
            state=self.game.snapshot(), parent=parent, action=action
        )
        parent.add_child(child)

        return child

    def update(self, leaf: Node, value: int) -> None:
        # Backpropagate the value up until root
        leaf.update_value(value)
        parent = leaf.parent

        while parent:
            parent.update_value(value)
            parent = parent.parent  # This will be `None` for root, so exit loop

    def train(self, maxiter: int = 100, strategy: SelectionStrategy = UCB) -> None:
        for _ in range(maxiter):
            # Select a node **starting from root** (see MCTS.select())
            parent = self.select(s=strategy)

            # Expand it (or just return the same node if terminal)
            leaf = self.expand(parent=parent)  # Can be terminal

            reward = self.game.rollout()

            # Backprop
            self.update(leaf=leaf, value=reward)

        return

    # def run_single(self, strategy: SelectionStrategy = Greedy):
    #     assert self._is_trained, "MCTS hasn't been trained yet"

    #     # Initial state always root
    #     node = self.root
    #     # Set game to match root state (inital state)
    #     self.game.set_state(node.game_state)
    #     print(node.game_state)

    #     while not self.game.is_finished:
    #         # Select a child according to strategy
    #         node = node.select(strategy=strategy)
    #         # Choose the action that led to the child, if possible
    #         action = (
    #             node.from_action
    #             if node.from_action in self.game.available_actions
    #             # Otherwise choose randomly
    #             else random.choice(list(self.game.available_actions))
    #         )
    #         # Play round
    #         self.game.play(action=action)
    #         # Print state
    #         print(self.game.snapshot())

    #     print(f"The winner is {self.game.check_winner()}")

    def best_action_value(self) -> Tuple[int, int, float]:
        best_child = self.root.select(Greedy)

        all_children = [(child.from_action, child.mean, child.n_visits) for child in self.root.children]

        return best_child.from_action, best_child.mean, all_children
    
    # def best_action_value(self, node)-> Tuple[int, float]:
    #     best_child = node.select(Greedy)
    #     return best_child.from_action, best_child.mean
