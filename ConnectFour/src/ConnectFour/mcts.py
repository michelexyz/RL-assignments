from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np

ACTIONS = (5, 7)


class ColumnFullError(Exception):
    pass


class ActionTuple(NamedTuple):
    player_action: int
    opp_action: int


class PlayerType(IntEnum):
    US = 1
    OPPONENT = 2


class GameBoard:
    grid: np.ndarray

    # TODO
    @property
    def is_finished(self) -> bool: ...  # Check if the matrix `self.grid` is full or not

    # TODO
    def __init__(self, nrows: int, ncols: int) -> None:
        self.grid = ...

    # TODO
    def game_result(self) -> int:
        assert self.is_finished
        return ...  # [1, 0, -1] depending on result

    # TODO
    def step(self, action: int, player: int) -> bool:
        action = self.validate_action(action)
        ...  # update the grid according to `action` and `player` type
        return self.is_finished

    # TODO
    def validate_action(self, action: int) -> int:
        if ...:  # Check whether `action` column of our `self.grid` has spots left
            raise ColumnFullError
        return action

    # TODO
    def available_actions_dict() -> Dict[str, int]:
        """
        Return a dictionary with the number of EMPTY positions per column (action)
        BE CAREFUL: Only include columns with spots left
        Like:
            RETURN THIS: {"5": 2, "7": 1}
            NOT THIS: {"5": 2, "7": 1, "2" 0, "1": 0 ...}
        """
        result = ...
        assert len(result) > 0
        assert all(left > 0 for left in result.values())
        ...

    def random_step(self, player: int) -> bool:
        # Get current available actions, return if there is none
        available = list(self.get_availabe_actions())
        if not available:
            return True
        return self.step(random.choice(available), player)

    def play(self, actions: List[ActionTuple]) -> Tuple[int, bool]:
        for p_action, opp_action in actions:
            self.step(action=p_action, player=PlayerType.US)
            self.step(action=opp_action, player=PlayerType.OPPONENT)

        if self.is_finished:
            # The node is terminal, so return
            return self.game_result(), True

        # The node that we set as leaf node is not terminal, so rollout random
        current_player = PlayerType.US
        while not self.is_finished:
            self.random_step(player=current_player)
            current_player = (
                PlayerType.US
                if current_player == PlayerType.OPPONENT
                else PlayerType.OPPONENT
            )
        return self.game_result(), False

    def get_availabe_actions(self) -> Set[int]:
        return GameBoard.get_availabe_actions(self.available_actions_dict())

    @staticmethod
    def update_available_actions(
        action: int, available_actions: Dict[str, int]
    ) -> Dict[str, int]:
        assert isinstance(action, int)
        available_actions[action] -= 1
        return {a: left for a, left in available_actions.items() if left > 0}

    @staticmethod
    def get_available_actions(available_actions_dict: Dict[str, int]) -> Set[int]:
        return set(int(a) for a, _ in available_actions_dict)


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
        if node.n_visits == 0:
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
    from_actions: Optional[ActionTuple]
    children: Set[Optional[Node]]
    depth: int
    n_visits: int
    value: float
    is_terminal: bool

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
        self.n_visits = 0
        self.value = 0.0
        self.is_terminal = False

    @classmethod
    def from_parent(cls, parent: Node, available_actions_dict: Dict[str, int]) -> Node:
        """Create a node from parent, expanding with random edge (action)"""
        # First player
        available = GameBoard.get_available_actions(available_actions_dict)
        unavailable = set(ch.from_actions.player_action for ch in parent.children)
        possible = list(set(ACTIONS) - unavailable)
        player_action = random.choice(possible)

        # Then opponent
        available = GameBoard.get_available_actions(
            GameBoard.update_available_actions(player_action, available_actions_dict)
        )
        opp_action = random.choice(list(available))
        return cls(
            _parent=parent,
            _from_actions=ActionTuple(
                player_action=player_action, opp_action=opp_action
            ),
        )

    def select(self, strategy: SelectionStrategy = UCB) -> Node:
        self.n_visits += 1
        if self.is_leaf:  # You only select when you have all your children
            return self
        return strategy.select(nodes=self.children)

    def __eq__(self, value: object) -> bool:
        """This will cause problems if either of the objects we are comparing
        is a root node. This should never be the case.
        """
        if isinstance(value, Node):
            return (
                self.depth == value.depth
                and self.from_actions.player_action == value.from_actions.player_action
            )
        return NotImplemented

    def __hash__(self) -> int:
        """This will cause problems if either of the objects we are comparing
        is a root node. This should never be the case.
        """
        return hash((self.depth, self.from_actions.player_action))


class MCTS:

    root: Node

    def __init__(self) -> None:
        self.root: Node = Node()

    def select(self) -> Node:
        node = self.root.select()
        while not node.is_leaf:
            node = node.select()
        return node

    def expand(self, parent: Node, available: Dict[str, int]) -> Node:
        if parent.is_terminal:
            return parent

        child = Node.from_parent(parent=parent, available_actions_dict=available)
        parent.children.add(child)
        return child

    def update(self, leaf: Node, value: int, is_terminal: bool) -> None:
        """Here is where the backprop happens"""
        leaf.is_terminal = is_terminal
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
        assert len(actions) > 1
        return actions[:0:-1]  # reverse and remove last action tuple (`None` always)


class Environment:
    def __init__(self) -> None:
        self.mcts = MCTS()
        self.game_board = GameBoard()

    def run(self, maxiter: int = 100):
        best_actions = []
        for _ in range(maxiter):
            leaf = self.mcts.expand(
                self.mcts.select(), available=self.game_board.available_actions_dict()
            )
            reward, is_terminal = self.game_board.play(
                actions=self.mcts.get_actions(leaf)
            )
            self.mcts.update(leaf=leaf, value=reward, is_terminal=is_terminal)

        ...
