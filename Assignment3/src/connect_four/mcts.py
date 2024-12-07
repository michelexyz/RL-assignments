import random
from typing import Dict

import numpy as np
from connect_four.game_board import GameBoard
from connect_four.node import UCB, Node, SelectionStrategy

QValuesDict = Dict[int, float]


class MCTS:

    root: Node | None
    game: GameBoard | None

    @property
    def qvalues(self) -> QValuesDict:
        return (
            {ch.from_action: ch.mean for ch in self.root.children}
            if self.root is not None
            else {}
        )

    def __init__(self) -> None:
        self.game = None
        self.root = None

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
        self.game.play(first_action=action)

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

    def run(
        self,
        game_state: np.ndarray,
        maxiter: int = 100,
        strategy: SelectionStrategy = UCB,
    ) -> QValuesDict:

        self.game = GameBoard.from_grid(game_state)
        self.root = Node.from_root(state=self.game.snapshot())

        for _ in range(maxiter):
            # Select a node **starting from root** (see MCTS.select())
            parent = self.select(s=strategy)

            # Expand it (or just return the same node if terminal)
            leaf = self.expand(parent=parent)  # Can be terminal

            reward = self.game.rollout()

            # Backprop
            self.update(leaf=leaf, value=reward)

        return self.qvalues


#     def best_action_value(self) -> Tuple[int, int, float]:
# best_child = self.root.select(Greedy)

# all_children = [
# (child.from_action, child.mean, child.n_visits)
# for child in self.root.children
# ]

# return best_child.from_action, best_child.mean, all_children
