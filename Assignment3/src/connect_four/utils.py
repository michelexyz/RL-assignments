from enum import IntEnum
from typing import Dict, Optional, Set

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class PlayerType(IntEnum):
    US = 1
    OPPONENT = 2


def available_actions(grid: np.ndarray) -> Set[int]:
    return set(np.where(grid == 0)[1])


def check_winner(grid: np.ndarray) -> Optional[PlayerType]:

    for player in [PlayerType.US, PlayerType.OPPONENT]:
        # Horizontal check
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1] - 3):
                if np.all([grid[row, col + i] == player for i in range(4)]):
                    return player

        # Vertical check
        for col in range(grid.shape[1]):
            for row in range(grid.shape[0] - 3):
                if np.all([grid[row + i, col] == player for i in range(4)]):
                    return player

        # Diagonal down (\) check
        for row in range(grid.shape[0] - 3):
            for col in range(grid.shape[1] - 3):
                if np.all([grid[row + i, col + i] == player for i in range(4)]):
                    return player

        # Diagonal up (/) check
        for row in range(3, grid.shape[0]):
            for col in range(grid.shape[1] - 3):
                if np.all([grid[row - i, col + i] == player for i in range(4)]):
                    return player
    return


def is_game_finished(grid: np.ndarray) -> bool:
    return (check_winner(grid) is not None) or (0 not in grid)


def display_plt(grid: np.ndarray, qvalues: Dict[int, str]) -> None:
    _, ax = plt.subplots(figsize=[7, 6])
    cmap = mcolors.ListedColormap(["white", "green", "blue"])
    norm = mcolors.BoundaryNorm([-1, 0.5, 1.5, 2.5], cmap.N)
    ax.matshow(grid, cmap=cmap, norm=norm)

    for x in range(8):
        ax.plot([x - 0.5, x - 0.5], [-0.5, 5.5], "k")
    for y in range(7):
        ax.plot([-0.5, 6.5], [y - 0.5, y - 0.5], "k")

    for a, value in qvalues.items():
        ax.text(
            a,
            -1,
            str(value),
            ha="center",
            va="center",
            fontsize=20,
            color="black",
        )

    ax.set_axis_off()
    plt.show()
    plt.close()


def display_circles(grid: np.ndarray, qvalues: Dict[int, str]) -> None:
    _, ax = plt.subplots(figsize=[7, 6])

    # Define colors for the pieces
    green_color = "#2ecc71"  # Emerald
    blue_color = "#3498db"  # Peter River

    # Draw the grid lines
    rows, cols = grid.shape
    for x in range(cols + 1):
        ax.plot([x - 0.5, x - 0.5], [-0.5, rows - 0.5], "k")
    for y in range(rows + 1):
        ax.plot([-0.5, cols - 0.5], [y - 0.5, y - 0.5], "k")

    # Plot each cell as a circle according to the board value
    # 0 = empty, 1 = green, 2 = blue
    for i in range(rows):
        for j in range(cols):
            val = grid[i, j]
            if val == 0:
                c = "white"
            elif val == 1:
                c = green_color
            elif val == 2:
                c = blue_color
            # Plot a circle marker (o) with a black edge for contrast
            ax.scatter(j, i, s=1200, c=c, marker="o", edgecolors="black")

    for a, value in qvalues.items():
        ax.text(
            a,
            -1,
            str(value),
            ha="center",
            va="center",
            fontsize=20,
            color="black",
        )

    # Adjust the aspect ratio and remove axes
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # invert y so top row is at index 0
    ax.set_axis_off()

    plt.show()
    plt.close()


def display_ascii(grid: np.ndarray, **kwargs) -> None:
    # ANSI color codes for the table
    GREEN = "\033[92m"  # Bright Green
    BLUE = "\033[96m"  # Bright Cyan
    GRAY = "\033[37m"  # Light Gray
    RESET = "\033[0m"  # Reset to default color

    # Symbols for each state
    symbols = {
        0: f"{GRAY}○{RESET}",
        PlayerType.US: f"{GREEN}●{RESET}",
        PlayerType.OPPONENT: f"{BLUE}●{RESET}",
    }

    # Print the board
    print(
        f"Player {PlayerType.US} playing {GREEN}●{RESET}\nPlayer {PlayerType.OPPONENT} playing {BLUE}●{RESET}"
    )
    for row in grid:
        print(" ".join(symbols[cell] for cell in row))
    print()  # Add an extra line for spacing
