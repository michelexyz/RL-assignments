import random
from typing import Dict

import numpy as np
from connect_four.game_board import GameBoard
from connect_four.node import UCB, Node, SelectionStrategy

QValuesDict = Dict[int, float]


class MCTS:

    root: Node | None
    game: GameBoard | None
    maxiter: int

    @property
    def qvalues(self) -> QValuesDict:
        return (
            {ch.from_action: ch.mean for ch in self.root.children}
            if self.root is not None
            else {}
        )

    def __init__(self, maxiter: int = 1048) -> None:
        self.game = None
        self.root = None
        self.maxiter = maxiter

    def select(self, s: SelectionStrategy) -> Node:
        """Returns either a LEAF or TERMINAL node"""
        # Start always from the top
        node = self.root.select(strategy=s)
        # Iterate through nodes until reaching a leaf / terminal node
        # update the game state to the node's state
        self.game.play(node.from_action)
        self.game.set_state(self.game.snapshot())
        

        while not (node.is_leaf or node.is_terminal):
            node = node.select(strategy=s)
            self.game.play(node.from_action)
            self.game.set_state(self.game.snapshot())

        # VERY important: set game state to start playing from
        #self.game.set_state(node.game_state)

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
        strategy: SelectionStrategy = UCB,
    ) -> QValuesDict:

        self.game = GameBoard.from_grid(game_state)
        self.root = Node.from_root(state=self.game.snapshot())

        for _ in range(self.maxiter):        

            # Select a node **starting from root** (see MCTS.select())
            parent = self.select(s=strategy)

            # Expand it (or just return the same node if terminal)
            leaf = self.expand(parent=parent)  # Can be terminal

            reward = self.game.rollout()

            # Backprop
            self.update(leaf=leaf, value=reward)

            self.game.reset()

        return self.qvalues
