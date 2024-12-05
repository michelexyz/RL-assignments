from connect_four.game_board import GameBoard
from connect_four.mcts import MCTS


class Environment:
    def __init__(self) -> None:
        self.mcts = MCTS()
        self.game_board = GameBoard()
        # Root state is of course initial board state
        self.mcts.root.game_state = self.game_board.snapshot()

    # TODO
    def run(self, maxiter: int = 100):
        for _ in range(maxiter):
            # Select a node **starting from root** (see MCTS.select())
            parent = self.mcts.select()

            # Expand it (or just return the same node if terminal)
            leaf = self.mcts.expand(parent, available=self.game_board.available_actions)

            # Play a game for the newly created node
            # If the node is terminal there is no problem, since action will be ignored
            reward, is_terminal = self.game_board.play(
                action=leaf.from_action, state=parent.game_state
            )
            # Update node game state
            leaf.game_state = self.game_board.snapshot()

            # Rollout if needed, from the state we left before (this is, after calling `play`)
            if not is_terminal:
                reward = self.game_board.rollout()

            # Backprop
            self.mcts.update(leaf=leaf, value=reward, is_terminal=is_terminal)

        ...  # I guess we want to return best actions to take?
