from typing import Set

from connect_four.node import Node


class MCTS:

    root: Node

    def __init__(self) -> None:
        self.root: Node = Node.from_root()

    def select(self) -> Node:
        # Start always from the top
        node = self.root.select()
        # Iterate through nodes until reaching a leaf node
        while not (node.is_leaf or node.is_terminal):
            node = node.select()
        return node

    def expand(self, parent: Node, available: Set[int]) -> Node:
        # Cancel expanding if node is terminal
        if parent.is_terminal:
            return parent
        child = Node.from_parent(parent=parent, available_actions=available)
        parent.add_child(child)
        return child

    def update(self, leaf: Node, value: int, is_terminal: bool) -> None:
        # Backpropagate the value up until root
        leaf.is_terminal = is_terminal
        leaf.value += value
        parent = leaf.parent
        while parent:
            parent.value += value
            parent = parent.parent  # This will be `None` for root, so exit loop
