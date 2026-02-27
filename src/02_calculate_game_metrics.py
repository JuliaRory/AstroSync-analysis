import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_hh_number(df_events):
    """подсчитывает среднее количество блокировок (домов отдыха) при взаимодействии со звездой"""

    n_interaction = 0  # кол-во взаимодействий
    n_hh = 0  # кол-во домов отдыха на одной звезде
    n_hh_all = []  # список с накоплением количеств блокировок 

    n_activate = df_events.loc[df_events.event == 'activate_star'].shape[0]
    gaze_verstion = True if n_activate != 0 else False
    target_event = 'activate_star' if gaze_verstion else 'select_star'
    for i in range(len(df_events)-1):  # идём по всем ивентам 
        # если звезда активировалась, а дальше начались пояса...
        if df_events['event'].iloc[i] == target_event and df_events['event'].iloc[i+1].find('step') != -1:  
            n_interaction += 1  # подсчитать новое взаимодействие
            n_hh_all.append(n_hh)  # сохранить кол-во домов отдыха для этого взаимодействия 
            n_hh = 0  # обнулить счётчик домов отдыха
        # если звезда активировалась, а дальше дом отдыха... 
        elif df_events['event'].iloc[i] == target_event and df_events['event'].iloc[i+1].find('holiday_home') != -1:
            n_hh += 1  # увеличить кол-во домов отдыха в копилочке 
    return np.mean(n_hh_all).round(2)  # вернуть среднее арифметическое от количества блокировок 


def calculate_p_green(df_events, score):  
    """метрики, связанные со взаимодействием со звездой (поясами)"""

    steps = [f'blast_step_{q}' for q in range(1, 9)] 
    gaze_version = True
    sel_events = df_events.loc[df_events.event == 'activate_star'].game_timestamp.values  # массив всех времён активации звёзд
    if len(sel_events) == 0:
        sel_events = df_events.loc[df_events.event == 'select_star'].game_timestamp.values  # массив всех времён активации звёзд 
        gaze_version = False
    
    df_center_steps = []  # табличка только с центральными поясами
    df_all_steps = []  # табличка со всеми поясами (включая внешние)

    time = 0
    n_stars = 0  # кол-во звёзд в игре
    for i in range(len(sel_events)+1)[1:]:  # цикл по всем кругам активации звезды
        curr_event = sel_events[i-1]  # время рассматриваемого ивента 
        next_event = sel_events[i] if i != len(sel_events) else df_events.game_timestamp.iloc[-1]  # время начала следующего (или конца игры)
        
        if df_events.loc[df_events.game_timestamp >= curr_event]['event'].iloc[1].find('step') == -1:
            time += 8  # 8 s - duration of a holiday home event
            continue  # skip it because it was a holiday home

        df_curr = df_events.loc[(df_events.game_timestamp >= curr_event) & (df_events.game_timestamp < next_event)]  # весь цикл взаимодействия со звездой
        df_steps = df_curr.loc[(df_curr.event.isin(steps)) | (df_curr.event == 'overkill_step')]  # all bands (взаимодействие + outer bands) 

        time += df_steps.shape[0]  # время взаимодействия со звездой (в с) от момента начала первого пояса до окончания взаимодействия (последнего outer band - удачного или нет)
        df_all_steps.append(df_steps) 
        
        central_steps = range(2, 8)  # центральные пояса для расчёта метрик
        if 'skipped_blast_step' in df_curr.event.values:
            central_steps = range(3, 8) # если пояс был пропущен, то начинать считать с третьего (т.к. фактически первый назван вторым)
        
        df_center_steps.append(df_curr.loc[df_curr.event.isin([f'blast_step_{q}' for q in central_steps])])
        
        n_stars += 1
    #print(len(df_center_steps))
    """только центральные пояса"""
    df_center_steps = pd.concat(df_center_steps, ignore_index=True)
    n_success = df_center_steps.loc[df_center_steps.decision == 'success'].shape[0]  # зелёные пояса
    p_green = round(n_success/ df_center_steps.shape[0] * 100, 2)  # процент зелёных поясов из всех центральных

    """все пояса"""
    df_all_steps = pd.concat(df_all_steps, ignore_index=True)
    
    n_good = df_all_steps.loc[df_all_steps.decision.isin(['success', 'no_overkill'])].shape[0]  # успешное заполнение поясов (либо зелёные внутри окружности, либо некрасные внешние)
    #n_bad_outer =  df_all_steps.loc[df_all_steps.decision.isin(['overkill'])].shape[0]  # плохие внешние
    n_bad_all =  df_all_steps.loc[df_all_steps.decision.isin(['overkill', 'failure'])].shape[0]  # плохие все (красные внутри и снаружи)

    # p_good = round(n_good / df_all_steps.shape[0] * 100, 2)  # процент хорошеньких поясов
    vel_good = round(n_good / (time/60), 2)  # скорость набирания хороших поясов (кол-во поясов в минуту)

    #score['vel_good']  = round((n_good - n_bad_outer)/ (time/60), 2)
    band_gain_min  = round((n_good - n_bad_all)/ (time/60), 2)  # скорость захватывания поясов (кол-во поясов в минуту)
    
    av_int_time = round(time / n_stars, 2)  # averaged interaction time

    score.update({
        'p_green_bands': p_green,  # percent of central green bands 
        # 'p_good_steps': p_good,
        'aver_time': av_int_time,  # averaged interaction time
        'green_steps_min': vel_good,  # the green bands per minute
        'band_gain_min': band_gain_min,  # the band gain per minute
        'n_stars': n_stars  # number of stars
    })

    return score


def calculate_green_steps(df_events):
    n_succ = df_events.loc[df_events.decision == 'success'].shape[0]
    velocity = float(round(n_succ/df_events.game_timestamp.iloc[-1]*1000*60, 2))
    return velocity


def calculate_overkill_number(df_events):
    """подсчитывает среднее количество неудачных внешних поясов при взаимодействии со звездой"""
    n_interaction = 0  # кол-во взаимодействий
    n_overkill = 0  # кол-во внешних задействованных неудачных поясов на одной звезде
    n_overkill_all = []   # список с накоплением количеств внешних неудачных поясов 

    for i in range(len(df_events)-1): # идём по всем ивентам 
        # если звезда взорвалась 
        if df_events['event'].iloc[i] == 'star_blasted': 
            n_interaction += 1  # подсчитать новое взаимодействие
            n_overkill_all.append(n_overkill)  # сохранить кол-во внешних поясов для этого взаимодействия 
            n_overkill = 0  # обнулить счётчик внешних поясов
        # если внешний НЕУДАЧНЫЙ пояс (удачный - no_overkill)
        elif df_events['event'].iloc[i] == 'overkill_step' and df_events['decision'].iloc[i] == 'overkill':
            n_overkill += 1  # увеличить кол-во внешних неудачных поясов в копилочке 

    return np.mean(n_overkill_all).round(2)


def calculate_game_metrics(df_events):
    """составляет словарь score со значениями метрик"""

    score = {}
    score['total_score'] = int(df_events.loc[df_events.event != 'change_score']['earned_points'].sum()) # total number of points in the condition or game
    # score['available_points'] = int(df_events.loc[df_events.event != 'change_score']['earned_points'].abs().sum())
    # if score['available_points'] != 0:
    #     score['relative_score'] = round(score['total_score'] / score['available_points'] * 100, 2)
    # else:
    #     score['relative_score'] = np.nan

    start_time = df_events.loc[df_events.event == 'start_game']['timestamp'].values
    end_time = df_events.loc[df_events.event == 'end_game']['timestamp'].values

    score['game_duration'] = round(int(sum((end_time - start_time) // 1000)) / 60, 2)
    # score['score_min'] = round(score['total_score'] / score['game_duration'], 2)
    # score['score_sec'] = round(score['total_score'] / score['game_duration'] / 60, 2)

    # score['green_steps_min'] = calculate_green_steps(df_events)
    
    score = calculate_p_green(df_events, score)  # calculate metrics connected with star interaction

    score['points_per_star'] = round(score['total_score'] / score['n_stars'], 2)  # averaged number of points per one star

    # score['n_hh'] = df_events.loc[df_events.event == 'start_holiday_home'].shape[0]  # number of holiday homes
    # score['n'] = df_events.loc[df_events.event == 'activate_star'].shape[0]  # number of star activations
    # score['hh_percent'] = round(score['n_hh'] / score['n'] * 100, 2)  # percent of holiday home 
    score['n_hh_average'] = calculate_hh_number(df_events)  

    score['n_overkill_average'] = calculate_overkill_number(df_events)
    return score


# def calculate_score(df):
#     total_score = int(df['total_points'].iloc[-1] - df['total_points'].iloc[0])
#     available_score = int(df.loc[df.event != 'change_score']['earned_points'].abs().sum())
#     relative_score = round(total_score / available_score * 100, 2) if available_score != 0 else np.nan
    
#     return relative_score


# def calculate_rel_score_progress(df_events, game_duration):
#     df_score = []
#     for min in np.arange(1, game_duration+1):
#         df_curr = df_events.loc[(df_events.game_timestamp >= (min-1) * 60 * 1000) & (df_events.game_timestamp < min * 60 * 1000)]
#         if 'activate_star' in df_curr.event.values:
#             df_score.append({'score': calculate_score(df_curr)})
#     return pd.DataFrame(df_score).T


filename_dataset = r'.\data\results\game_dataset.csv'
filename_dataset = r'.\data\results\OPM_results\game_dataset.csv'
df = pd.read_csv(filename_dataset)

df_score = []  # табличка для записи значений метрик по играм
df_score_cond = []  # табличка для записи значений метрик по условиям
# df_progress = []



for subject in tqdm(df.subject.unique()):
    if subject == 'test1':
        for mode in ['im', 'qm']:
            """Подсчёт метрик отдельно по играм"""
            for game in [1, 2, 3]:
                df_events = df.loc[(df.subject == subject) & (df['mode'] == mode) & (df.n_game == game)]

                if df_events.shape[0] == 0: # если игры нет (такой случай должен быть один - 04AB, режим im, игра 3)
                    print(f'{subject}_{mode}_{game}_{df_events.shape}')
                    continue
                
                score = calculate_game_metrics(df_events)
                score['condition'] = mode 
                score['n_game'] = game
                score['subject'] = subject
                df_score.append(score)

            """Подсчёт метрик по условию"""
            df_events = df.loc[(df.subject == subject) & (df['mode'] == mode)]

            score = calculate_game_metrics(df_events)
            score['condition'] = mode 
            score['subject'] = subject
            df_score_cond.append(score)

        df_score = pd.DataFrame(df_score)
        df_score_cond = pd.DataFrame(df_score_cond)
        # df_progress = pd.concat(df_progress, ignore_index=True)

        folder_output = r'.\data\results\OPM_results'
        df_score.to_csv(os.path.join(folder_output, 'game_metrics_per_n_game.csv'), index=False)
        df_score_cond.to_csv(os.path.join(folder_output, 'game_metrics_per_condition.csv'), index=False)
        # df_progress.to_csv(os.path.join(folder_output, 'score_progress.csv'), index=False)

    elif subject == 'test2':
        """Подсчёт метрик отдельно по играм"""
        df_score = []
        for game in [1, 2, 3]:
            
            df_events = df.loc[(df.subject == subject) & (df.n_game == game)]
            if df_events.shape[0] == 0: # если игры нет (такой случай должен быть один - 04AB, режим im, игра 3)
                print(f'{subject}_{mode}_{game}_{df_events.shape}')
                continue
            
            score = calculate_game_metrics(df_events)
            score['condition'] = mode 
            score['n_game'] = game
            score['subject'] = subject
            df_score.append(score)
        df_score = pd.DataFrame(df_score)

        folder_output = r'.\data\results\OPM_results'
        df_score.to_csv(os.path.join(folder_output, 'game_metrics_im_tests.csv'), index=False)

    elif subject == 'test3_8ch':
        """Подсчёт метрик отдельно по играм"""
        df_score = []
        game = 1
            
        df_events = df.loc[(df.subject == subject) & (df.n_game == game)]
        
        if df_events.shape[0] == 0: # если игры нет (такой случай должен быть один - 04AB, режим im, игра 3)
            print(f'{subject}_{mode}_{game}_{df_events.shape}')
            continue
        
        score = calculate_game_metrics(df_events)
        score['condition'] = mode 
        score['n_game'] = game
        score['subject'] = subject
        df_score.append(score)
        df_score = pd.DataFrame(df_score)

        folder_output = r'.\data\results\OPM_results'

        df_score.to_csv(os.path.join(folder_output, 'game_metrics_8ch.csv'), index=False)

    elif subject == 'test4_eeg_opm':
        """Подсчёт метрик отдельно по методам регистрации сигнала"""
        df_score = []
        for method in ['eeg', 'opm']:
            df_events = df.loc[(df.subject == subject) & (df['mode'] == method)]
            
            for game in [1, 2, 3]:
                df_game = df_events.loc[df_events.n_game == game]
                if df_game.shape[0] == 0: # если игры нет (такой случай должен быть один - 04AB, режим im, игра 3)
                    print(f'{subject}_{mode}_{game}_{df_game.shape}')
                    continue
                
                score = calculate_game_metrics(df_game)
                score['condition'] = method 
                score['n_game'] = game
                score['subject'] = subject
                score['filename'] = df_game.filename.unique()
                df_score.append(score)
        df_score = pd.DataFrame(df_score)

        folder_output = r'.\data\results\OPM_results'
        df_score.to_csv(os.path.join(folder_output, 'game_metrics_eeg_opm_tests.csv'), index=False)


# for subject in tqdm(df.subject.unique()):
    
#     for mode in ['im', 'qm']:
#         """Подсчёт метрик отдельно по играм"""
#         for game in [1, 2, 3]:
#             df_events = df.loc[(df.subject == subject) & (df['mode'] == mode) & (df.n_game == game)]

#             if df_events.shape[0] == 0: # если игры нет (такой случай должен быть один - 04AB, режим im, игра 3)
#                 print(f'{subject}_{mode}_{game}_{df_events.shape}')
#                 continue
            
#             score = calculate_game_metrics(df_events)
#             score['condition'] = mode 
#             score['n_game'] = game
#             score['subject'] = subject
#             df_score.append(score)

#             # progress = calculate_rel_score_progress(df_events, score['game_duration'])
#             # progress['mode'] = mode
#             # progress['n_game'] = game
#             # progress['subject'] = subject
#             # df_progress.append(progress)

#         """Подсчёт метрик по условию"""
#         df_events = df.loc[(df.subject == subject) & (df['mode'] == mode)]

#         score = calculate_game_metrics(df_events)
#         score['condition'] = mode 
#         score['subject'] = subject
#         df_score_cond.append(score)

# df_score = pd.DataFrame(df_score)
# df_score_cond = pd.DataFrame(df_score_cond)
# # df_progress = pd.concat(df_progress, ignore_index=True)

# folder_output = r'.\data\results\'
# df_score.to_csv(os.path.join(folder_output, 'game_metrics_per_n_game.csv'), index=False)
# df_score_cond.to_csv(os.path.join(folder_output, 'game_metrics_per_condition.csv'), index=False)
# # df_progress.to_csv(os.path.join(folder_output, 'score_progress.csv'), index=False)
