from enum import IntEnum

from mcts import MCTS

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class GameType(IntEnum):
    HUMAN_VS_RANDOM = 0
    MCTS_VS_RANDOM = 1
    MCTS_VS_HUMAN = 2




def game_play(game_type: GameType):
    


    display_actions = [
        {'index': 0, 'val': 'X'}, 
        {'index': 1, 'val': 'X'}, 
        {'index': 2, 'val': 'X'}, 
        {'index': 3, 'val': 'X'}, 
        {'index': 4, 'val': 'X'}, 
        {'index': 5, 'val': 'X'}, 
        {'index': 6, 'val': 'X'}
    ]


    mcts = MCTS()
    game = mcts.game
    human_intervention = False

    

    done = False
    while not done:

        mcts.run(maxiter=2048)

        best_action , best_mean, all_actions = mcts.best_action()

        for 
        
        for action in best_action[1]:
            display_actions[action.action]['val'] = round(action.qval, 2)
        display(game.state, display_actions)
        act = best_action[0].action
        if human_intervention:
            try:
                act = int(input("Choose a column (0-6): "))
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 6.")
            
        reward, done = game.step(act)
        if done:
            if reward > 0:
                print('Win')
            elif reward < 0:
                print('Lose')
            else:
                print('Draw')
            display(game.state, display_actions)




def display(board, values):
    fig, ax = plt.subplots(figsize=[7, 6])
    cmap = mcolors.ListedColormap(['white', 'green', 'blue'])
    norm = mcolors.BoundaryNorm([-1, 0.5, 1.5, 2.5], cmap.N)
    ax.matshow(board, cmap=cmap, norm=norm)
    
    for x in range(8):
        ax.plot([x - .5, x - .5], [-.5, 5.5], 'k')
    for y in range(7):
        ax.plot([-.5, 6.5], [y - .5, y - .5], 'k')
        
    for v in values:
        ax.text(v['index'], -1, str(v['val']), ha='center', va='center', fontsize=20, color='black')
        
    ax.set_axis_off()
    plt.show()
    plt.close()