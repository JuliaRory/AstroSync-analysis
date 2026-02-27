import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_timestamp(line, which='Pygame', next_word='SPEED', divide=1):
    """Определить время по одной из систем отчёта, задаваемых which"""
    find_word = f'{which}_timestamp'
    if next_word != 'nothing':
        t = line[line.find(find_word)+len(find_word)+2:line.find(next_word)-2]
    else:
        t = line[line.find(find_word)+len(find_word)+2:-2]

    return int(t) // divide


def find_value(line, word, next_word):
    """Определяет из строки искомое значение для параметра word, 
        вырезая его из строки до слова next_word"""
    value = line[line.find(word) + len(word):line.find(next_word)]
    return int(value)


def find_position(line, next_word='Pygame'):
    """Найти позицию звезды (координаты х и у) из строки"""
    word = 'Position'
    position = line[line.find(word)+len(word)+2:line.find(next_word)-2]
    x = position[1:position.find(',')]
    y = position[position.find(',')+2:-1]
    return (int(x), int(y))


def define_n_star(df_events):
    """Определяет порядковый номер звезды"""
    sel_events = df_events.loc[df_events.event == 'activate_star'].game_timestamp.values

    star_pos = []
    for event in sel_events:
        star_pos.append(df_events.loc[df_events.game_timestamp == event][['pos_x', 'pos_y']].iloc[0])

    star_pos = np.unique(np.array(star_pos), axis=0)
    n_star = 0
    df_events['n_star'] = -1
    for pos in star_pos:
        df_events.loc[(df_events.pos_x == pos[0]) & (df_events.pos_y == pos[1]), 'n_star'] = n_star
        n_star += 1

    return df_events


def create_dataset(filename, subject, add_time=0, add_points=0):
    """Пробегает по файлу и создаёт на его основе пандас-табличку с событиями в игре"""
    df_events = []
    points = 0
    if subject == '01TG':
        which, next_word = 'SPEED', 'EyeLink'
    elif subject.find('test') != -1:
        which, next_word = 'EyeLink', 'VX'
    elif int(subject[:2]) >= 18:
        which, next_word = 'EyeLink', 'VX'
    else:
        which, next_word = 'EyeLink', 'nothing'

    divide = 1 if not subject in ['02ES', '03AC'] else 1000000

    with open(filename, 'r') as file:

        for line in file:
            
            if line.find('Game started') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                df_events.append({'event': 'start_game', 'timestamp': timestamp, 
                                  'pos_x': 0, 'pos_y': 0, 'res_timestamp': res_timestamp,
                                  'decision': '-', 'earned_points': 0, 'total_points': 0})
            
            elif line.find('star_selected') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                pos = find_position(line)
                df_events.append({'event': 'select_star', 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': '-', 'earned_points': 0, 'total_points': points})
            
            elif line.find('star_unselected') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                pos = find_position(line)
                df_events.append({'event': 'unselect_star', 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': '-', 'earned_points': 0, 'total_points': points})
            
            elif line.find('star_activated') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                pos = find_position(line, next_word='Clf')
                df_events.append({'event': 'activate_star', 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': '-', 'earned_points': 0, 'total_points': points})
            
            elif line.find('holiday_home.') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                pos = find_position(line, next_word='Triangle')
                df_events.append({'event': 'start_holiday_home', 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': '-', 'earned_points': 0, 'total_points': points})
            
            elif line.find('holiday_home_end') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                pos = find_position(line)
                df_events.append({'event': 'end_holiday_home', 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': '-', 'earned_points': 0, 'total_points': points})
            
            elif line.find('blast_step') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                if line.find('skipped_empty') != -1:
                    event = 'skipped_blast_step'
                    pos = find_position(line)
                    decision = '-'
                    earned_points = 0
                else:
                    event = line[line.find('blast_step'):line.find('. P')]
                    pos = find_position(line, next_word='Clf')
                    decision = line[line.find('Decision:')+len('Decision:')+1:line.find('. Stars')]
                    earned_points = find_value(line, word='Score_change: ', next_word='. Pygame')
                points += earned_points
                df_events.append({'event': event, 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': decision,'earned_points': earned_points, 'total_points': points})

            elif line.find('overkill_step') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                pos = find_position(line, next_word='Clf')
                
                if line.find('no overkill') != -1:
                    decision = 'no_overkill'
                    earned_points = 0
                else:
                    decision = 'overkill'
                    earned_points = find_value(line, word='Score_change: ', next_word='. Pygame')
                points += earned_points
                df_events.append({'event': 'overkill_step', 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': decision,'earned_points': earned_points, 'total_points': points})
            
            elif line.find('star_blasted') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                earned_points = 0
                df_events.append({'event': 'star_blasted', 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': '-','earned_points': earned_points, 'total_points': points})
                

            elif line.find('score_changed') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                earned_points = int(line[line.find("Earned points")+len('Earned points')+2:line.find('. Total')])
                total_points = line[line.find("Total points")+len('Total points')+2:line.find('. Pygame')]
                #points += earned_points
                df_events.append({'event': 'change_score', 'timestamp': timestamp, 
                                  'pos_x': pos[0], 'pos_y': pos[1], 'res_timestamp': res_timestamp,
                                  'decision': '-','earned_points': earned_points, 'total_points': points})
            
            elif line.find('Game ended ') != -1 or line.find('Game was quitted!') != -1:
                timestamp = find_timestamp(line, which='Pygame', next_word='SPEED')
                res_timestamp = find_timestamp(line, which=which, next_word=next_word, divide=divide)
                df_events.append({'event': 'end_game', 'timestamp': timestamp, 
                                  'pos_x': 0, 'pos_y': 0, 'res_timestamp': res_timestamp,
                                  'decision': '-','earned_points': 0, 'total_points': points})
    
    n_games = 1
    df_events = pd.DataFrame(df_events)

    """Сложные игры с переопределением времени"""
    pause_1_time = df_events.loc[df_events.event == 'start_game']['timestamp'].iloc[0]
    df_events['game_timestamp'] = df_events['timestamp'].values - pause_1_time + add_time

    ind_end_field_1 = df_events.loc[df_events.event == 'end_game'].index[0]
    dur_field_1 = df_events.loc[df_events.event == 'end_game']['game_timestamp'].iloc[0]

    if df_events.loc[df_events.event == 'start_game']['timestamp'].shape[0] == 2:  # if two games
        n_games = 2
        pause_2_time = df_events.loc[df_events.event == 'start_game']['timestamp'].iloc[1]

        df_events.loc[df_events.index > ind_end_field_1, 'game_timestamp'] += (dur_field_1 - (pause_2_time - pause_1_time))

    mode = 'im' if filename.find('im_log') != -1 else 'qm'
    if filename.find('eeg_log') != -1:
        mode = 'eeg'
    if filename.find('opm_log') != -1:
        mode = 'opm'
    df_events['mode'] = mode

    game = filename[filename.find('log_game')+len('log_game')+1:filename.find('.txt')]
    df_events['n_game'] = int(game)

    df_events['subject'] = subject
    df_events['total_points'] += add_points
    if add_time != 0:
        n_games = 2 # это для контроля половинчатых игр, если 2 - игра полная, иначе - 1
    return df_events, n_games 

"""main AstroSync experiment"""
folder_input = r'.\data\raw\exp'
folder_output = r'.\data\results'
filename_output = 'game_dataset.csv'


"""OPM AstroSync tests"""
folder_input = r'.\data\raw\OPM-tests'
folder_output = r'.\data\results\OPM_results'
filename_output = 'game_dataset.csv'

n = 2
add_time = 0
add_points = 0
df_dataset = []
for subject in tqdm(os.listdir(folder_input)): 

    for cond in ['im', 'qm', 'eeg', 'opm']:
        n_game = 1
        for filename in os.listdir(os.path.join(folder_input, subject)):
            if filename.find('nouse') != -1: # игра, которую надо пропустить
                continue
            if filename.find('log_game') != -1 and filename.find('test') == -1 and filename.find(cond) != -1:
                
                # print(filename)
                df, n = create_dataset(os.path.join(folder_input, subject, filename), subject, add_time, add_points)
                
                df['filename'] = filename[:filename.find('.')]
                df['n_game'] = n_game
                df = define_n_star(df) # определяет порядковый номер звёзд 
                
                if (n == 1):
                    # это людишки, у которых реально только половинки игры (одно поле) в режимах, поэтому не надо объединять игры
                    if ((subject == '20EC') & (cond == 'qm')) | ((subject == '24EK') & (cond == 'im')) | (subject == 'test2') | (subject == 'test4_eeg_opm'):
                        add_time = 0
                        add_points = 0
                        n_game += 1
                    # для всех остальных людей если есть половинки, то имеется в виду, что их надо объединить в одну игру
                    else:
                        add_time = df.game_timestamp.iloc[-1]
                        add_points = df.total_points.iloc[-1]
                else:
                    add_time = 0
                    add_points = 0
                    n_game += 1 # это номер игры (1 2 3)
                if subject == 'test2':
                    print(df)
                df_dataset.append(df)

df_dataset = pd.concat(df_dataset, ignore_index=True)
df_dataset.to_csv(os.path.join(folder_output, filename_output), index=False)
