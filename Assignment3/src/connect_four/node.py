from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Set

import numpy as np
from connect_four.utils import available_actions


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

    game_state: np.ndarray
    parent: Optional[Node]
    from_action: Optional[int]

    children: Set[Node]
    depth: int
    n_visits: int
    value: float

    MAX_CHILDREN: int = 2

    @property
    def is_leaf(self) -> bool:
        return True if self.available_actions else False

    @property
    def is_terminal(self) -> bool:
        return 0 not in self.game_state

    @property
    def available_actions(self) -> Set[int]:
        return available_actions(self.game_state) - self.get_children_actions()

    @property
    def mean(self) -> float:
        if self.n_visits == 0:
            return 0
        return self.value / self.n_visits

    def __init__(
        self,
        game_state: np.ndarray,
        parent: Optional[Node] = None,
        from_action: Optional[int] = None,
    ) -> None:
        self.game_state = game_state
        self.parent = parent
        self.from_action = from_action

        self.children = set()
        self.depth = parent.depth + 1 if parent else 0
        self.n_visits = 0
        self.value = 0.0

    @classmethod
    def from_root(cls, state: np.ndarray) -> Node:
        return cls(game_state=state)

    @classmethod
    def from_parent(cls, state: np.ndarray, parent: Node, action: int) -> Node:
        return cls(game_state=state, parent=parent, from_action=action)

    def select(self, strategy: SelectionStrategy = UCB) -> Node:
        if (
            self.is_leaf or self.is_terminal
        ):  # You only select when you have all your children
            return self
        assert self.children
        return strategy.select(nodes=self.children)

    def add_child(self, child: Node) -> None:
        # Check that a parent doesn't have two kids with the same action
        assert child.from_action not in [ch.from_action for ch in self.children]

        self.children.add(child)
        # Check that we don't have more children than permitted
        assert not (len(self.children) > self.MAX_CHILDREN)

    def get_children_actions(self) -> Set[int]:
        return set(ch.from_action for ch in self.children)

    def update_value(self, value: int) -> None:
        self.value += value
        self.n_visits += 1
