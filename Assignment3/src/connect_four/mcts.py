import random
from typing import Tuple

import numpy as np
from connect_four.game_board import GameBoard
from connect_four.node import Greedy, Node

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

    def __init__(self) -> None:
        self.game = GameBoard.from_grid(game_init)
        self.root = Node.from_root(state=self.game.snapshot())

    def select(self) -> Node:
        """Returns either a LEAF or TERMINAL node"""
        # Start always from the top
        node = self.root.select()
        # Iterate through nodes until reaching a leaf / terminal node
        while not (node.is_leaf or node.is_terminal):
            node = node.select()

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

        action = random.choice(list(parent.available_actions))
        self.game.play(action=action)

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

    def run(self, maxiter: int = 100) -> None:
        for _ in range(maxiter):
            # Select a node **starting from root** (see MCTS.select())
            parent = self.select()

            # Expand it (or just return the same node if terminal)
            leaf = self.expand(parent=parent)  # Can be terminal

            reward = self.game.rollout()

            # Backprop
            self.update(leaf=leaf, value=reward)
        return

    def best_action_value(self) -> Tuple[int, float]:
        best_child = self.root.select(Greedy)
        return best_child.from_action, best_child.mean
