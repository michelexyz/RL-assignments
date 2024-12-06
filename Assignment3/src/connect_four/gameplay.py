from enum import IntEnum

from mcts import MCTS

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from connect_four.utils import PlayerType
import time

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


    done = False
    while not done:

        mcts.run(maxiter=2048)

        best_action , best_mean, all_actions = mcts.best_action()

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
        else:
            act = best_action
       
        if game_type in [GameType.MCTS_VS_RANDOM , GameType.HUMAN_VS_RANDOM]:# if we are playing against random agent we step for both players
            game.play(act)

        elif game_type == GameType.MCTS_VS_HUMAN:

            #time.sleep(3)
            #print("computer is thinking")
            game.step(act, PlayerType.US)

            for action in display_actions.keys():
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

          
        if game.is_finished:

            reward = game.game_result()

            match reward:
                case 1:
                    print('Win')
                case 0:
                    print('Lose')
                case -1:
                    print('Draw')

            
            for action in display_actions.keys():

                display_actions[action]['val'] = 'X'
            
            display(game.state, display_actions)
        else:
            mcts = MCTS(game_state=game.snapshot())




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