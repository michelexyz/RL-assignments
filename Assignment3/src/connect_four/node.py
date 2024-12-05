from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Optional, Set

import numpy as np


class SelectionStrategy(ABC):
    @staticmethod
    @abstractmethod
    def strategy(node: Node) -> float:
        pass

    @classmethod
    def select(cls, nodes: Set[Node]) -> Node:
        return max(nodes, key=cls.strategy)


class UCB(SelectionStrategy):
    C = 2

    @staticmethod
    def strategy(node: Node) -> float:
        if not node.n_visits > 0:
            return math.inf
        return node.mean + (
            UCB.C * math.sqrt(math.log(node.parent.n_visits / node.n_visits))
        )


class Greedy(SelectionStrategy):
    @staticmethod
    def strategy(node: Node) -> float:
        return node.mean


class Node:

    parent: Optional[Node]
    from_action: Optional[int]

    children: Set[Node]
    depth: int
    n_visits: int
    value: float
    is_terminal: bool
    game_state: Optional[np.ndarray]

    MAX_CHILDREN: int = 2

    @property
    def is_leaf(self) -> bool:
        return len(self.children) < self.MAX_CHILDREN

    @property
    def mean(self) -> float:
        if self.n_visits == 0:
            return 0
        return self.value / self.n_visits

    def __init__(
        self,
        parent: Optional[Node] = None,
        from_action: Optional[int] = None,
    ) -> None:
        """
        Initializes a Node, either as a root node or via a class methods

        Args:
            parent (Optional[Node]): The parent node. None if this is the root node.
            from_action (Optional[int]): The action that led to this node from its parent.
        """
        self.parent = parent
        self.from_action = from_action

        self.children = set()
        self.depth = parent.depth + 1 if parent else 0
        self.n_visits = 0
        self.value = 0.0
        self.is_terminal = False
        self.game_state = None

    @classmethod
    def from_root(cls) -> Node:
        return cls()

    @classmethod
    def from_parent(cls, parent: Node, available_actions: Set[int]) -> Node:
        # Available actions in the game board are given as argument
        # Make sure to not create a child that comes from the same action
        # (type of edge) as other child from the parent
        assert available_actions, "No actions provided for `from_parent` method"

        avail = list(available_actions - set(ch.from_action for ch in parent.children))
        assert avail, "No actions left to take"

        return cls(parent=parent, from_action=random.choice(avail))

    def select(self, strategy: SelectionStrategy = UCB) -> Node:
        self.n_visits += 1
        if self.is_leaf:  # You only select when you have all your children
            return self
        return strategy.select(nodes=self.children)

    def add_child(self, child: Node) -> None:
        # Check that a parent doesn't have two kids with the same action
        assert child.from_action not in [ch.from_action for ch in self.children]

        self.children.add(child)
        # Check that we don't have more children than permitted
        assert not (len(self.children) > self.MAX_CHILDREN)
