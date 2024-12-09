import argparse
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from connect_four.game import Game, GameType
from connect_four.utils import display_ascii


@dataclass
class Args:
    from_empty: bool
    game_type: Literal["mcts_vs_random", "human_vs_mcts", "human_vs_random"]
    mcts_iter: int


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


def parse_args(args: List[str]) -> Args:
    parser = argparse.ArgumentParser(
        prog="connect_four",
        description="Playing the Connect Four game using Monte Carlo Tree Search",
    )

    parser.add_argument(
        "--from_empty",
        action="store_true",
        help="Wheter to start from an empty game board",
    )
    parser.add_argument(
        "--game_type",
        choices=["mcts_vs_random", "human_vs_mcts", "human_vs_random"],
        default="mcts_vs_random",
        help="Type of game to play. Default: '%(default)s'",
    )
    parser.add_argument(
        "--mcts_iter",
        type=int,
        default=1048,
        help="Maximum number of MCTS iterations. Default: %(default)s",
    )

    args = parser.parse_args()

    return Args(
        from_empty=args.from_empty, game_type=args.game_type, mcts_iter=args.mcts_iter
    )


def choose_game_type(game_type: str) -> GameType:
    match game_type:
        case "mcts_vs_random":
            return GameType.MCTS_VS_RANDOM
        case "human_vs_mcts":
            return GameType.HUMAN_VS_MCTS
        case "human_vs_random":
            return GameType.HUMAN_VS_RANDOM


def run(args: Args) -> None:
    game_state = game_init if not args.from_empty else None
    game_type = choose_game_type(args.game_type)
    game = Game(game_state=game_state, game_type=game_type, mcts_maxiter=args.mcts_iter)
    game.play(display_fun=display_ascii)
