import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import warnings
warnings.filterwarnings('ignore')


def redefine_timestamps(timestamps):
    time_list = []
    for time in timestamps:
        curr_time = time[0]//1000000
        time_list.extend(np.arange(curr_time-40, curr_time, 2))
    return np.array(time_list, dtype=np.int64)


def define_coords(all_coords, timestamps, timestamp_end, duration=1000):
    return all_coords[np.where((timestamps > timestamp_end-duration) & (timestamps < timestamp_end))[0]]


def calculate_deviation(coords_gaze, pos_x, pos_y, corr_x=0, corr_y=0):
    x_dev = coords_gaze[:, 0]+corr_x - pos_x
    y_dev = coords_gaze[:, 1]+corr_y - pos_y
    
    dev = np.sqrt(x_dev ** 2 + y_dev ** 2)
    return np.nanmedian(dev).round(2)


def calculate_variance(coords):
    return np.nanvar(coords).round(2)


def calculate_range(coords):
    return (np.nanmax(coords) - np.nanmin(coords)).round(2)


def calculate_path(x, y):
    x_diff = np.diff(x).reshape((-1, 1))
    y_diff = np.diff(y).reshape((-1, 1))
    return np.sum(np.sqrt(x_diff ** 2 + y_diff ** 2)).round(2)


def calculate_features(step, timestamp, coords, timestamps,end_time,  duration=1000):
    curr_coords = define_coords(coords, timestamps, timestamp, duration)
    
    metrics= ['deviation', 'gaze_pos_x', 'gaze_pos_y', 'var_x', 'var_y', 'var']
    p_nan = np.isnan(curr_coords[:, 0]).sum() / len(curr_coords) * 100
    step['p_nan'] = p_nan
    if len(curr_coords) == 0 or p_nan == 100:
        for name in metrics:
            step[name] = np.nan
    else:
        x, y = curr_coords[:, 0], curr_coords[:, 1]
        x = x[np.where((x > np.nanpercentile(x, 2.5)) & (x < np.nanpercentile(x, 97.5)))]
        y = y[np.where((y > np.nanpercentile(y, 2.5)) & (y < np.nanpercentile(y, 97.5)))]
        if len(x) == 0 or len(y) == 0:
            for name in metrics:
                step[name] = np.nan
        else:
            step['deviation'] = calculate_deviation(curr_coords, step['pos_x'], step['pos_y'])
            step['gaze_pos_x'] = np.nanmedian(curr_coords[:, 0]).round(2)
            step['gaze_pos_y'] = np.nanmedian(curr_coords[:, 1]).round(2)
            step['var_x'] = calculate_variance(x)
            step['var_y'] = calculate_variance(y)
            step['var'] = np.mean([step['var_x'], step['var_y']])
            step['range_x'] = calculate_range(x)
            step['range_y'] = calculate_range(y)
            step['range_mean'] = np.mean([step['range_x'], step['range_y']])
            step['range_max'] = np.max([step['range_x'], step['range_y']])
            step['range'] = np.sqrt(step['range_x'] ** 2 + step['range_y'] ** 2)
            step['gaze_path'] = calculate_path(curr_coords[:, 0], curr_coords[:, 1])

            center_coords = define_coords(coords, timestamps, end_time+300, 300)
            step['av_x'] = np.nanmedian(center_coords[:, 0]).round(2)
            step['av_y'] = np.nanmedian(center_coords[:, 1]).round(2)
            step['norm_dev'] = calculate_deviation(curr_coords, step['av_x'], step['av_y'])

    return step


filename_dataset = r'.\data\results\game_dataset.csv'
df = pd.read_csv(filename_dataset)
df.rename(columns={'mode': 'condition'}, inplace=True)

steps = ['activate_star', 'overkill_step'] + [f'blast_step_{i}' for i in range(1, 9)]

df_corr = pd.read_csv(r'.\data\results\corrections.csv')

folder_output = r'.\data\results\gaze_features'
counter = 0
for subject in tqdm(df.subject.unique()):

    counter += 1
    if not subject in ['03AC', '07TS', '13AU', '14BE', '15AZ', '21EC', '25PP']: # RIGHT EYE
        continue
    # if subject in ['03AC', '07TS', '13AU', '14BE', '15AZ', '21EC', '25PP']: # LEFT EYE
    #     continue
    
    df_stars_all, df_activ_all = [], []
    for condition in ['im', 'qm']:

        for game in [1, 2, 3]:
  
            df_events = df.loc[(df.subject == subject) & (df['condition'] == condition) & (df.n_game == game)]

            if df_events.shape[0] == 0: # если игры нет (такой случай должен быть один - 04AB, режим im, игра 3)
                print(f'{subject}_{condition}_{game}_{df_events.shape}')
                continue
            
            filenames_log = df_events.filename.unique()
            for filename_log in filenames_log:
                filename_rec = f'{condition}_rec_game_{filename_log[-1:]}.hdf'

                try:
                    with h5py.File(os.path.join(r'.\data\raw\exp', subject, filename_rec), 'r') as h5f:
                        coords = (h5f['eyeData/data'][:-1])[:, 2:] # RIGHT EYE
                        # coords = (h5f['eyeData/data'][:-1])[:, :2] # LEFT EYE
                        # coords = coords[:, 2:] if sum(np.isnan(coords[:, 0])) == len(coords) else coords[:, :2]
                        
                        timestamps = redefine_timestamps(h5f['eyeData/blocks'][:])
                except:
                    print(subject, filename_rec)
                    continue
                
                df_corr_curr = df_corr.loc[(df_corr.subject == subject) & (df_corr.filename == filename_log)]
                if len(df_corr_curr) == 0:
                    print('no correction coeffient', subject, filename_log)
                    continue
                corr_x, corr_y = df_corr_curr[['corr_x', 'corr_y']].values[0]

                df_subj = df.loc[(df.subject == subject) & (df.filename == filename_log)].reset_index()

                star_ind = df_subj.loc[df_subj.event == 'activate_star'].index.to_list()
                
                # interaction

                n = 1  # counter of stars (by order of activation)
                df_stars = []
                for ind in star_ind:  # all star activation (blasting and holiday home)
                    
                    if df_subj.iloc[ind+1].event.find('step') != -1:  # drop holiday home
                        n_star = df_subj.iloc[ind]['n_star']
                        df_curr = df_subj.iloc[ind:ind+13].copy()
                        df_curr = df_curr.loc[(df_curr.n_star == n_star) & (df_curr.event.isin(steps))]
                        n_blast = df_curr.loc[df_curr.event.isin([f'blast_step_{i}' for i in range(1, 9)])].shape[0]
                        n_outer = df_curr.loc[df_curr.event == 'overkill_step'].shape[0]
                        start_time = df_curr.loc[df_curr.event == 'activate_star']['res_timestamp'].values[0] - 2000
                        
                        for i in range(500, 2000+(df_curr.shape[0]-1)*1000+1, 500):
                            wind = {}
                            wind['n_blast'] = n_blast
                            wind['n_outer'] = n_outer
                            wind['window'] = i
                            if i <= 2000:
                                wind['step'] = 'activation'
                            elif i <= 2000 + n_blast*1000:
                                wind['step'] = 'blast'
                            else:
                                wind['step'] = 'overkill'
                            for name in ['n_star', 'subject', 'filename', 'condition', 'n_game', 'pos_x', 'pos_y']:
                                wind[name] = df_curr[name].unique()[0]

                            wind = calculate_features(wind, start_time+i, coords, timestamps, end_time=start_time+2000, duration=500)

                            df_stars.append(wind)
                        n += 1

                df_stars_all.append(pd.DataFrame(df_stars))

    df_stars_all = pd.concat(df_stars_all, ignore_index=True)
    df_stars_all.to_csv(os.path.join(folder_output, f'{subject}_steps.csv'), index=False)
            
 


# df_stars_all = pd.concat(df_stars_all, ignore_index=True)
# df_stars_all.to_csv(os.path.join(folder_output, 'gaze_steps.csv'), index=False)

# df_activ_all = pd.concat(df_activ_all, ignore_index=True)
# df_activ_all.to_csv(os.path.join(folder_output, 'gaze_activ.csv'), index=False)