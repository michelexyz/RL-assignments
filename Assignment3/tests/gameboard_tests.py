import random
import unittest
import numpy as np
from typing import Optional

from connect_four.utils import PlayerType, check_winner


def generate_random_grid():
    grid = np.zeros((6, 7), dtype=int)
    for _ in range(random.randint(10, 30)):  # Randomly add 10-30 moves
        row = random.randint(0, 5)
        col = random.randint(0, 6)
        grid[row, col] = random.choice([PlayerType.US, PlayerType.OPPONENT])
    return grid

class TestCheckWinner(unittest.TestCase):

    def test_horizontal_win(self):
        grid = np.zeros((6, 7), dtype=int)
        grid[0, 0:4] = PlayerType.US
        self.assertEqual(check_winner(grid), PlayerType.US)

    def test_vertical_win(self):
        grid = np.zeros((6, 7), dtype=int)
        grid[0:4, 0] = PlayerType.OPPONENT
        self.assertEqual(check_winner(grid), PlayerType.OPPONENT)

    def test_diagonal_down_win(self):
        grid = np.zeros((6, 7), dtype=int)
        for i in range(4):
            grid[i, i] = PlayerType.US
        self.assertEqual(check_winner(grid), PlayerType.US)

    def test_diagonal_up_win(self):
        grid = np.zeros((6, 7), dtype=int)
        for i in range(4):
            grid[3 - i, i] = PlayerType.OPPONENT
        self.assertEqual(check_winner(grid), PlayerType.OPPONENT)

    def test_no_winner(self):
        grid = np.zeros((6, 7), dtype=int)
        self.assertIsNone(check_winner(grid))

    def test_random_configs(self):
        for _ in range(100):  # Generate 100 random configurations
            grid = generate_random_grid()
            winner = check_winner(grid)
            if winner is not None:  # If there's a winner, print the grid
                print("\nWinning grid:")
                print(grid)
                print(f"Winner: {winner}")

            

if __name__ == '__main__':
    unittest.main()