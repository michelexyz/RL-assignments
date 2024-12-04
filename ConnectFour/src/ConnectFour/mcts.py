from __future__ import annotations

import math
import random
from typing import List, NamedTuple, Optional, Set

ACTIONS = (5, 7)


class ActionTuple(NamedTuple):
    player_action: int
    opp_action: int


class GameBoard:
    pass


class UCB:
    C = 2

    @staticmethod
    def ucb(node: Node) -> float:
        if node.n_visits == 0:
            return math.inf
        return (node.value / node.n_visits) + (
            UCB.C * math.sqrt(math.log(node.parent.n_visits / node.n_visits))
        )

    @staticmethod
    def select(nodes: Set[Node]) -> Node:
        return max(nodes, key=UCB.ucb)


class Node:

    parent: Optional[Node]
    from_actions: Optional[ActionTuple]
    children: Set[Optional[Node]]
    depth: int
    n_visits: int
    value: float

    MAX_CHILDREN: int = 2

    @property
    def is_leaf(self) -> bool:
        return len(self.children) < self.MAX_CHILDREN

    def __init__(
        self,
        _parent: Optional[Node] = None,
        _from_actions: Optional[ActionTuple] = None,
    ) -> None:
        """This class is supposed to be initialized as:
            - `Node()`, when creating a **root** node
            - `Node.from_parent(parent)`, when a **non-root** note is created
        That's why arguments in the constructor start with a `_`.
        """
        self.parent = _parent
        self.from_actions = _from_actions

        self.children = set()
        self.depth = _parent.depth + 1 if _parent else 0
        self.n_visits: int = 0
        self.value: float = 0.0

    @classmethod
    def from_parent(cls, parent: Node) -> Node:
        """Create a node from parent, expanding with random edge (actions)"""
        unavailable = set(ch.from_actions.player_action for ch in parent.children)
        possible = set(ACTIONS) - unavailable
        return cls(
            _parent=parent,
            _from_actions=ActionTuple(
                player_action=random.choice(possible), opp_action=random.choice(ACTIONS)
            ),
        )

    def select(self) -> Node:
        self.n_visits += 1
        if self.is_leaf:
            return self
        return UCB.select(nodes=self.children)

    def __eq__(self, value: object) -> bool:
        """This will cause problems if either of the objects we are comparing
        is a root node. This should never be the case.
        """
        if isinstance(value, Node):
            return self.depth == value.depth and self.from_action == value.from_action
        return NotImplemented

    def __hash__(self) -> int:
        """This will cause problems if either of the objects we are comparing
        is a root node. This should never be the case.
        """
        return hash((self.depth, *self.from_actions))


class MCTS:

    root: Node

    def __init__(self) -> None:
        self.root: Node = Node()

    def select(self) -> Node:
        node = self.root.select()
        while not node.is_leaf:
            node = node.select()
        return node

    def expand(self, parent: Node) -> Node:
        child = Node.from_parent(parent=parent)
        parent.children.add(child)
        return child

    def update(self, leaf: Node, value: int) -> None:
        leaf.value += value
        parent = leaf.parent
        while parent:
            parent.value += value
            parent = parent.parent

    def get_actions(self, leaf: Node) -> List[ActionTuple]:
        actions = [leaf.from_actions]
        parent = leaf.parent
        while parent:  # iterate until reaching root
            actions.append(parent.from_actions)
            parent = parent.parent
        return actions[
            1::-1
        ]  # reverse list to take actions in order from root, excluding root `from_action`


class Environment:
    def __init__(self) -> None:
        self.mcts = MCTS()
        self.game_board = GameBoard()

    def run(self):
        leaf = self.mcts.expand(self.mcts.select())
        reward = self.game_board.play(actions=self.mcts.get_actions(leaf))
        self.mcts.update(leaf=leaf, value=reward)
