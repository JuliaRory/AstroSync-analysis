import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import warnings
warnings.filterwarnings('ignore')
import json

# функция для доставания глазодвигательных событий (фиксаций, саккад и морганий)
def get_event(event):
    mes = str(event)[2:-1]
    res_new = mes.replace('\\n', '')
    res_new = res_new.replace(' ', '')
    res_new = res_new.replace("\\", ' ')
    event = json.loads(res_new)
    return event

# функция для создания из набора сообщений о глазодвигательных событиях датасета
def parse_event(event):
    event_type = event['type'][:event['type'].find('_')]
    if event_type == 'blink':
        event_parsed = {
            'event': event_type, 
            'time_start': event['start_time'],
            'time_end': event['end_time']
        }
    elif event_type == 'saccade':
        try:
            event_parsed = {
                'event': event_type, 
                'time_start': event['start_time'],
                'time_end': event['end_time'],
                'x_start': event['gaze']['start']['x'],
                'y_start': event['gaze']['start']['y'],
                'x_stop': event['gaze']['stop']['x'],
                'y_stop': event['gaze']['stop']['y'],
            }
        except:  # just a very strange bug
            event_parsed = {
                'event': event_type, 
                'time_start': event['start_time'],
                'time_end': event['end_time'],
                'x_start': event['gaze']['start']['x'],
                'y_start': event['gaze']['start']['y'],
                'x_stop':  np.nan,
                'y_stop': np.nan,
            }
    else:
        event_parsed = {
            'event': event_type, 
            'time_start': event['start_time'],
            'time_end': event['end_time'],
            'x_average': event['gaze']['average']['x'],
            'y_average': event['gaze']['average']['y']
        }
    return event_parsed


def create_df_events(filename):
    with h5py.File(filename, 'r') as h5f:
        df_fix = pd.DataFrame({
            'time_start': h5f['A/Events/Efix/start'][:][0],
            'time_end': h5f['A/Events/Efix/end'][:][0],
            'x_average': h5f['A/Events/Efix/posX'][:][0],
            'y_average': h5f['A/Events/Efix/posY'][:][0],
        }).round(2)
        df_fix['event'] = 'fixation'
        
        df_sac = pd.DataFrame({
            'time_start': h5f['A/Events/Esacc/start'][:][0],
            'time_end': h5f['A/Events/Esacc/end'][:][0],
            'x_start': h5f['A/Events/Esacc/posX'][:][0],
            'y_start': h5f['A/Events/Esacc/posY'][:][0],
            'x_stop': h5f['A/Events/Esacc/posXend'][:][0],
            'y_stop': h5f['A/Events/Esacc/posYend'][:][0],
        }).round(2)
        df_sac['event'] = 'saccade'
    return pd.concat([df_fix, df_sac], ignore_index=True)


def find_rel_fixation(step, df_events, duration=2000):

    end = step['res_timestamp']
    df_curr = df_events.loc[((df_events.res_start > end - duration) & (df_events.res_end < end)) | \
                            ((df_events.res_start <= end - duration) & (df_events.res_end > end - duration)) | \
                            ((df_events.res_start <= end) & (df_events.res_end > end))]
    for metric in ['interaction_type', 'res_timestamp', 'condition', 'n_game', 'subject', 'filename', 'n_star', 'n']:
        df_curr[metric] = step[metric]
    
    return df_curr


filename_dataset = r'.\data\results\game_dataset.csv'
df = pd.read_csv(filename_dataset)
df.rename(columns={'mode': 'condition'}, inplace=True)

steps = ['activate_star', 'overkill_step'] + [f'blast_step_{i}' for i in range(1, 9)]



folder_output = r'.\data\results\gaze_features'
counter = 0
for subject in tqdm(df.subject.unique()):
    
    df_fix_all =  []
    for condition in ['im', 'qm']:

        for game in [1, 2, 3]:
            df_events = df.loc[(df.subject == subject) & (df['condition'] == condition) & (df.n_game == game)]

            if df_events.shape[0] == 0: # если игры нет (такой случай должен быть один - 04AB, режим im, игра 3)
                print('no game presented', f'{subject}_{condition}_{game}_{df_events.shape}')
                continue

            filenames_log = df_events.filename.unique()
            for filename_log in filenames_log:
                
                filename_rec = f'{condition}_rec_game_{filename_log[-1:]}.hdf'
                try:
                    with h5py.File(os.path.join(r'.\data\raw\exp', subject, filename_rec), 'r') as h5f:
                        raw_events = h5f['eyeEvents/messages'][:]
                
                    events = [get_event(event[2]) for event in raw_events]
                    timestamps = [int(event[0]//1000000) for event in raw_events]
                    # print(events)
                    df_events = [parse_event(event) for event in events if event['type'].find('start') == -1]
                    df_events = pd.DataFrame(df_events).round()

                    timestamps = [timestamp for i, timestamp in enumerate(timestamps) if events[i]['type'].find('start') == -1]
                    df_events['res_end'] = timestamps
                    df_events['diff'] = df_events['res_end'] - df_events['time_end']

                    diff = df_events['diff'].mean().round()

                except:
                    print('cannot open rec file', subject, filename_rec)
                    continue
                
                try:
                    filename_rec = f'{condition}_EyeLink_game_{filename_log[-1:]}_reparsed.h5'
                    df_events = create_df_events((os.path.join(r'.\data\raw\exp', subject, filename_rec)))
                    df_events['res_start'] = df_events['time_start'] + diff
                    df_events['res_end'] = df_events['time_end'] + diff
                except:
                    print('cannot open eyelink file', subject, filename_rec)
                    continue
                
                

                df_subj = df.loc[(df.subject == subject) & (df.filename == filename_log)].reset_index()

                star_ind = df_subj.loc[df_subj.event == 'activate_star'].index.to_list()
                
                # activation

                n = 1  # counter of stars (by order of activation)
                n_hh, hh_pos = 0, -1  # counter of holiday homes
                df_activ = [[] for _ in range(8)]

                for ind in star_ind:  # all star activation (blasting and holiday home)

                    n_star = df_subj.iloc[ind]['n_star']
                    if (n_hh != 0) & (hh_pos == n_star):
                        # print(filename, ind, n_star, 'double selection')
                        continue
                    step = df_subj.iloc[ind].copy()
                    
                    if df_subj.iloc[ind+1].event.find('step') != -1:
                        inter_type = 'blast'
                        n_hh, hh_pos = 0, -1
                    else:
                        inter_type = 'holiday_home'
                        n_hh += 1
                        hh_pos = df_subj.iloc[ind].n_star
                    
                    step['interaction_type'] = inter_type
                    step['n'] = n

                    df_fix = find_rel_fixation(step, df_events)

                    n += 1

                    df_fix_all.append(pd.DataFrame(df_fix))

    counter += 1

    df_fix_all= pd.concat(df_fix_all, ignore_index=True)
    df_fix_all.to_csv(os.path.join(folder_output, f'{subject}_events.csv'), index=False)
