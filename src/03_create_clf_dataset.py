import os
import numpy as np
import pandas as pd


def find_proba(line):
    clf_output = line[line.find('[')+1:line.find(']')].split(', ')
    return [float(value) for value in clf_output[-10:]]


columns_names = ['action'] + [f'act_{i}' for i in range(10)] + [f'step_{i}' for i in range(80)] + [f'overkill_{i}' for i in range(40)] 


def create_clf_dataset(filename, subject, n_add):
    df_stars = []
    n = 0
    with open(filename, 'r') as file:
        for line in file:

            if line.find('star_activated') != -1:
                overkill_counter = 0
                proba_array = find_proba(line)
            
            elif line.find('holiday_home.') != -1:
                proba_array.extend([float(value) for value in np.full(12*10, np.nan)])
                proba_array.insert(0, 'holiday_home')
                df_stars.append(dict(zip(columns_names, proba_array)))
            
            elif line.find('skipped_empty_blast_step') != -1:
                proba_array.extend([float(value) for value in np.full(10, np.nan)])

            elif line.find(' blast_step') != -1:
                proba_array.extend(find_proba(line))
            
            elif line.find('overkill_step') != -1:
                overkill_counter += 1
                proba_array.extend(find_proba(line))

            elif line.find('star_blasted') != -1:
                if overkill_counter != 4:
                    proba_array.extend([float(value) for value in np.full(10*(4-overkill_counter), np.nan)])
                proba_array.insert(0, 'star_blasted')
                df_stars.append(dict(zip(columns_names, proba_array)))
            
            elif line.find('Game ended ') != -1 or line.find('Game was quitted!') != -1:
                n += 1

    df_stars = pd.DataFrame(df_stars)

    mode = 'im' if filename.find('im_log') != -1 else 'qm'
    df_stars['mode'] = mode

    game = filename[filename.find('log_game')+len('log_game')+1:filename.find('.txt')]
    df_stars['n_game'] = int(game)

    df_stars['subject'] = subject

    if n_add == 1:
        n = 2
    return df_stars, n


folder_input = r'.\data\raw\exp'
folder_output = r'.\data\results'

df_dataset = []
for subject in os.listdir(folder_input):
    for cond in ['im', 'qm']:
        n_game = 1
        n = 0
        for filename in os.listdir(os.path.join(folder_input, subject)):
            if filename.find('log_game') != -1 & filename.find('test') == -1 and filename.find(cond) != -1:
                df, n = create_clf_dataset(os.path.join(folder_input, subject, filename), subject, n)
                df['n_game'] = n_game

                if n == 2: # if there were 2 fields
                    n_game += 1 # change number of game

                df_dataset.append(df)
                
    # for filename in os.listdir(os.path.join(folder_input, subject)):
    #     if filename.find('log_game') != -1 & filename.find('test') == -1:
    #         df = create_clf_dataset(os.path.join(folder_input, subject, filename), subject)
    #         df_dataset.append(df)

df_dataset = pd.concat(df_dataset, ignore_index=True)
df_dataset.to_csv(os.path.join(folder_output, 'clf_dataset.csv'), index=False)