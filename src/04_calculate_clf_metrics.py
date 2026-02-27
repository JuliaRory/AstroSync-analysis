import os
import numpy as np
import pandas as pd
import copy


# def find_latency(inds, queue=2, delay=0):
#     if len(inds) == 0:  # if no values than latency is equal to zero
#         latency = 0
#     else: 
#         inds_diff = np.diff(inds)  # find step between indexes
#         latency = inds[-1]  # latency is the last index
#         for i in range(len(inds_diff[queue:])):
#             if np.array_equal(np.full(queue, 1), inds_diff[i-queue:i]):
#                 latency = inds[i]
#                 break
#         latency -= delay # if there are a delay, do not count it 
#         latency *= 100 # in ms
#     return latency


# def find_max_dur_and_n_onset(motor_inds, queue=3):
#     inds_diff = np.diff(motor_inds)
#     dur, max_dur, n_onset = 0, 0, 1
#     for i in range(len(inds_diff[queue:])):
#         if np.array_equal(np.full(queue, 1), inds_diff[i-queue:i]):
#             dur += 1
#         else:
#             n_onset += 1
#             if dur >= max_dur:
#                 max_dur = copy.deepcopy(dur)
#                 dur = 0
#     if dur >= max_dur:
#         max_dur = copy.deepcopy(dur)
#     max_dur *= 100
#     return max_dur, n_onset


def find_latency(proba, queue, thr=.5, desired_state='motor'):
    cond = lambda x: x >= thr if desired_state == 'motor' else x < thr
    latency = 0
    for i in range(len(proba))[queue:]:
        av_p = np.mean(proba[i-queue:i])
        if cond(av_p): 
            latency = i * 100
            break
    return latency


def find_max_dur_and_n_onset(proba, queue=3, thr=.5):
    dur, max_dur, n_onset = 0, 0, 0
    curr_state = 'REST'
    for i in range(len(proba))[queue:]:
        av_p = np.mean(proba[i-queue:i])
        if av_p >= thr:
            if curr_state == 'REST':
                curr_state = 'MOTOR'
                n_onset += 1
            dur += 1
        else:
            if curr_state == 'MOTOR':
                curr_state = 'REST'
            if dur >= max_dur:
                max_dur = copy.deepcopy(dur)
                dur = 0
    if dur >= max_dur:
        max_dur = copy.deepcopy(dur)
    return max_dur * 100, n_onset



def calculate_clf_metrics(df_stars, thr = 0.5, queue = 3):
    act, steps, overkill = [f'act_{i}' for i in range(10)], [f'step_{i}' for i in range(80)], [f'overkill_{i}' for i in range(40)] 
    df_score = []
    for star_ind in df_stars.loc[df_stars.action == 'star_blasted'].index:
        score = {}

        star = df_stars.iloc[star_ind]
        n_step1_nan = np.isnan(np.array([float(value) for value in star[steps].values])).sum() # 10 or 0 
        n_steps = len(steps) - n_step1_nan # 70 or 80
        n_overkill = len(overkill) - np.isnan(np.array([float(value) for value in star[overkill].values])).sum() # 0, 10, 20, 30
        score['A'] = round((sum(star[steps] >= thr) + sum(star[overkill] < thr)) / (n_steps + n_overkill), 2)
        score['A_motor'] = round(sum(star[steps[n_step1_nan+10:-10]] >= thr) / (n_steps-20), 2) # only central ones
        score['proba_mean'] = round(np.mean(star[steps[n_step1_nan+10:-10]]), 2)

        # motor_inds = np.where((star[steps] >= thr) == 1)[0]  # where after star activation values are above thr
        # score['latency_motor'] = find_latency(motor_inds, queue, n_step1_nan)

        # stop_inds = np.where((star[steps] < thr) == 1)[0]
        # score['latency_stop_forward'] = find_latency(stop_inds, queue)
        # stop_inds = np.array([stop_inds[i] for i in range(len(stop_inds)-1, -1, -1)])
        # score['latency_stop_backward'] = find_latency(stop_inds, queue)

        latency_motor = find_latency(star[act+steps], queue=5, thr=.5, desired_state='motor') - len(act)*100
        if latency_motor > 0:
            latency_motor -= n_step1_nan
        score['latency_motor'] = latency_motor

        score['latency_stop_forward'] = find_latency(star[overkill], queue=5, thr=.5, desired_state='rest')
        proba = star[steps].values
        proba_inverse =  np.array([proba[i] for i in range(len(proba)-1, -1, -1)])
        score['latency_stop_backward'] = find_latency(proba_inverse, queue=5, thr=.5, desired_state='rest')
        
        score['stop_shift'] = score['latency_stop_forward'] if score['latency_stop_forward'] < score['latency_stop_backward'] else -score['latency_stop_backward']

        score['stop_shift2'] = find_latency(star[steps[-10:]+overkill], queue=5, thr=.5, desired_state='rest') - 10*100

        # score['max_dur'], score['n_onset'] = find_max_dur_and_n_onset(motor_inds, queue=3)
        score['max_dur'], score['n_onset'] = find_max_dur_and_n_onset(star[act[-queue:]+steps], queue=3, thr=.5)

        df_score.append(score)
    df_score = pd.DataFrame(df_score)
    df_score['n_star'] = np.arange(len(df_score))
    return df_score


def add_subject_info(df, mode, game, subject):
    df['mode'] = mode
    df['game'] = game
    df['subject'] = subject
    return df


filename_dataset = r'.\data\results\clf_dataset.csv'
df = pd.read_csv(filename_dataset)
df_score_progress = []
df_score_average = []
for subject in df.subject.unique():
    for mode in ['im', 'qm']:
        for game in [1, 2, 3]:
            df_events = df.loc[(df.subject == subject) & (df['mode'] == mode) & (df.n_game == game)].copy().reset_index()
            score = calculate_clf_metrics(df_events)
            score = add_subject_info(score, mode, game, subject)
            df_score_progress.append(score)

            score_average = score.mean(numeric_only=True)
            score_average.drop('n_star', inplace=True)
            score_average = add_subject_info(score_average, mode, game, subject)
            df_score_average.append(score_average)

df_score_progress = pd.concat(df_score_progress, ignore_index=True)
df_score_average = pd.DataFrame(df_score_average)

folder_output = r'.\data\results'
df_score_progress.to_csv(os.path.join(folder_output, 'clf_metrics_progress.csv'), index=False)
df_score_average.to_csv(os.path.join(folder_output, 'clf_metrics_average.csv'), index=False)