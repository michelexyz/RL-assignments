from enum import IntEnum

from connect_four.mcts import MCTS
from connect_four.game_board import GameBoard

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from connect_four.utils import PlayerType
import time

class GameType(IntEnum):
    HUMAN_VS_RANDOM = 0
    MCTS_VS_RANDOM = 1
    MCTS_VS_HUMAN = 2




def game_play(game_type: GameType = GameType.HUMAN_VS_RANDOM, empty_start: bool = False, maxiter= 2048):
    


    display_actions = [
        {'index': 0, 'val': 'X'}, 
        {'index': 1, 'val': 'X'}, 
        {'index': 2, 'val': 'X'}, 
        {'index': 3, 'val': 'X'}, 
        {'index': 4, 'val': 'X'}, 
        {'index': 5, 'val': 'X'}, 
        {'index': 6, 'val': 'X'}
    ]

    if empty_start:

        initial_game = GameBoard()
        mcts = MCTS(game_state=initial_game.snapshot())

    else:
        mcts = MCTS()

    initial_game = mcts.game



    game = GameBoard.from_grid(initial_game.snapshot())
    
    done = False
    while not done:
        print("Thinking...")
        mcts.train(maxiter=maxiter)

        best_action , best_mean, all_actions = mcts.best_action_value()

        for (from_action, mean, n_visits) in all_actions:
            display_actions[from_action]['val'] = round(mean, 2)
        
        display(game.snapshot(), display_actions)
        if game_type == GameType.HUMAN_VS_RANDOM:
            
            try:
                act = int(input(f"Choose an available column: game.available_actions: {game.available_actions}"))

                if act not in game.available_actions:
                    raise ValueError
            except ValueError:
                print(f"{act} is Invalid input. Please enter a number between 0 and 6.")
        else: # mcts as green player
            act = best_action
       
        if game_type in [GameType.MCTS_VS_RANDOM , GameType.HUMAN_VS_RANDOM]:# if we are playing against random agent we step for both players
            game.play(act)

        elif game_type == GameType.MCTS_VS_HUMAN:

            game.step(act, PlayerType.US)

            for action in range(len(display_actions)):
                if action not in game.available_actions:

                    display_actions[action]['val'] = 'X'
                else:
                    display_actions[action]['val'] = '?'

            display(game.snapshot(), display_actions)
            if not game.is_finished:

                try:
                
                    opponent_act = int(input(f"Choose an available column: game.available_actions: {game.available_actions}"))

                    if opponent_act not in game.available_actions:
                        print(f"{opponent_act} is Invalid input. Please enter a number between 0 and 6.")

                        raise ValueError
                        
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 6.")
                    continue

                opponent_act.step(act, PlayerType.OPPONENT)

          
        if game.is_finished: # if finished show results

            reward = game.game_result()

            match reward:
                case 1:
                    print('Green wins')
                case 0:
                    print('Blue wins')
                case -1:
                    print('Draw')

            
            for action in range(len(display_actions)):

                display_actions[action]['val'] = 'X'
            
            display(game.snapshot(), display_actions)

            done = True
        else:
            mcts = MCTS(game_state=game.snapshot())




# def display(board, values):
#     fig, ax = plt.subplots(figsize=[7, 6])
#     cmap = mcolors.ListedColormap(['white', 'green', 'blue'])
#     norm = mcolors.BoundaryNorm([-1, 0.5, 1.5, 2.5], cmap.N)
#     ax.matshow(board, cmap=cmap, norm=norm)
    
#     for x in range(8):
#         ax.plot([x - .5, x - .5], [-.5, 5.5], 'k')
#     for y in range(7):
#         ax.plot([-.5, 6.5], [y - .5, y - .5], 'k')
        
#     for v in values:
#         ax.text(v['index'], -1, str(v['val']), ha='center', va='center', fontsize=20, color='black')
        
#     ax.set_axis_off()
#     plt.show()
#     plt.close()


def display(board, values):
    fig, ax = plt.subplots(figsize=[7, 6])
    
    # Define colors for the pieces
    green_color = '#2ecc71'  # Emerald
    blue_color = '#3498db'   # Peter River
    
    # Draw the grid lines
    rows, cols = board.shape
    for x in range(cols + 1):
        ax.plot([x - 0.5, x - 0.5], [-0.5, rows - 0.5], 'k')
    for y in range(rows + 1):
        ax.plot([-0.5, cols - 0.5], [y - 0.5, y - 0.5], 'k')
    
    # Plot each cell as a circle according to the board value
    # 0 = empty, 1 = green, 2 = blue
    for i in range(rows):
        for j in range(cols):
            val = board[i, j]
            if val == 0:
                c = 'white'
            elif val == 1:
                c = green_color
            elif val == 2:
                c = blue_color
            # Plot a circle marker (o) with a black edge for contrast
            ax.scatter(j, i, s=1200, c=c, marker='o', edgecolors='black')
    
    # Plot the text values below the board, if any
    for v in values:
        ax.text(v['index'], -1, str(v['val']), ha='center', va='center', 
                fontsize=20, color='black')
    
    # Adjust the aspect ratio and remove axes
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # invert y so top row is at index 0
    ax.set_axis_off()
    
    plt.show()
    plt.close()
