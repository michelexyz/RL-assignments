from __future__ import annotations

import math
import random
from typing import List, Optional, Set

ACTIONS = (5, 7)


class GameBoard:
    pass


class UCB:
    C = 2

    @staticmethod
    def ucb(node: Node) -> float:
        return (node.value / node.n_visits) + (
            UCB.C * math.sqrt(math.log(node.parent.n_visits / node.n_visits))
        )

    @staticmethod
    def select(nodes: Set[Node]) -> Node:
        return max(nodes, key=UCB.ucb)


class Node:

    parent: Optional[Node]
    from_action: Optional[int]
    children: Set[Optional[Node]]
    depth: int

    MAX_CHILDREN: int = 2

    @property
    def is_leaf(self) -> bool:
        return len(self.children) < self.MAX_CHILDREN

    def __init__(
        self, _parent: Optional[Node] = None, _from_action: Optional[int] = None
    ) -> None:
        self.parent = _parent
        self.from_action = _from_action

        self.children = set()
        self.depth = _parent.depth + 1 if _parent else 0
        self.n_visits: int = 0
        self.value: float = 0.0

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Node):
            return self.depth == value.depth and self.from_action == value.from_action
        return NotImplemented

    @classmethod
    def from_parent(cls, parent: Node) -> Node:
        """Create a node from parent, expanding with random edge (action)"""

        possible = set(ACTIONS) - set([ch.from_action for ch in parent.children])
        return cls(_parent=parent, _from_action=random.choice(possible))

    def select(self) -> Node:
        ...
        return UCB.select(nodes=self.children)

    def get_actions(self) -> List[int]:
        actions = [self.from_action]
        parent = self.parent
        while parent:  # iterate until reaching root
            actions.append(parent.from_action)
            parent = parent.parent
        return actions[
            1::-1
        ]  # reverse list to take actions in order from root, excluding root `from_action`

    def update(self, value: int) -> None:
        self.value += value
        parent = self.parent
        while parent:
            parent.value += value
            parent = parent.parent


class MCTS:

    root: Node

    def __init__(self) -> None:
        self.root: Node = ...

    # def init()

    def select(self) -> Node:
        if self.root.is_leaf:
            node = self.expand(self.root)
        else:
            node = self.root

        while not node.is_leaf:
            node = node.select()
        return node

    def expand(self, parent: Node) -> Node:
        child = Node.from_parent(parent=parent)
        parent.children.add(child)
        return child

    def rollout(self, leaf: Node, game: GameBoard) -> int:
        leaf.update(game.play(actions=leaf.get_actions()))
