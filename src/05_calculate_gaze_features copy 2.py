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


def calculate_deviation(coords_gaze, pos_center, corr_x=0, corr_y=0):
    x_dev = coords_gaze[:, 0]+corr_x - pos_center[0]
    y_dev = coords_gaze[:, 1]+corr_y - pos_center[1]
    
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


def calculate_features(step, coords, timestamps, corr_x, corr_y, end_time, duration=1000):
    curr_coords = define_coords(coords, timestamps, step['res_timestamp'], duration)
    
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
            step['deviation'] = calculate_deviation(curr_coords, step[['pos_x', 'pos_y']])
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

            center_coords = define_coords(coords, timestamps, end_time+300, 300)
            step['av_x'] = np.nanmedian(center_coords[:, 0]).round(2)
            step['av_y'] = np.nanmedian(center_coords[:, 1]).round(2)
            step['norm_dev'] = calculate_deviation(curr_coords, step[['av_x', 'av_y']])
            
            step['gaze_path'] = calculate_path(curr_coords[:, 0], curr_coords[:, 1])

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
    # if not subject in ['03AC', '07TS', '13AU', '14BE', '15AZ', '21EC', '25PP']: # RIGHT EYE
    #     continue
    if subject in ['03AC', '07TS', '13AU', '14BE', '15AZ', '21EC', '25PP']: # LEFT EYE
        continue
    
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
                        # coords = (h5f['eyeData/data'][:-1])[:, 2:] # RIGHT EYE
                        coords = (h5f['eyeData/data'][:-1])[:, :2] # LEFT EYE
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
                
                # activation

                n = 1  # counter of stars (by order of activation)
                n_hh, hh_pos = 0, -1  # counter of holiday homes
                df_activ = []

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
                    step = calculate_features(step, coords, timestamps, corr_x, corr_y, end_time=step['res_timestamp'], duration=2000)
                            
                    df_activ.append(step)
                    n += 1

                df_activ = pd.DataFrame(df_activ)
                df_activ_all.append(df_activ)
           
   
                # interaction


                n = 1  # counter of stars (by order of activation)
                df_stars = []
                for ind in star_ind:  # all star activation (blasting and holiday home)
                    
                    if df_subj.iloc[ind+1].event.find('step') != -1:  # drop holiday home
                        n_star = df_subj.iloc[ind]['n_star']
                        df_curr = df_subj.iloc[ind:ind+13].copy()
                        df_star = df_curr.loc[(df_curr.n_star == n_star) & (df_curr.event.isin(steps))].reset_index(drop=True)
                        for i in range(df_star.shape[0]):  # all steps 
                            step = df_star.iloc[i].copy()

                            if step['event'] == 'activate_star':
                                step_name = 'activation'
                                end_time = step['res_timestamp']
                                dur = 2000
                            else:
                                step_name = f'blast_step_{i}' if step['event'].find('blast_step') != -1 else f'overkill_step_{i}'
                                dur = 1000
                            step['step'] = step_name  # to numerate steps for plots
                            step['n_step'] = i + 1
                            step['n'] = n

                            step = calculate_features(step, coords, timestamps, corr_x, corr_y, end_time=end_time, duration=dur)
                            
                            df_stars.append(step)
                        n += 1

                df_stars = pd.DataFrame(df_stars)
                df_stars_all.append(df_stars)


    df_activ_all = pd.concat(df_activ_all, ignore_index=True)
    df_activ_all.to_csv(os.path.join(folder_output, f'{subject}_activ.csv'), index=False)

    df_stars_all = pd.concat(df_stars_all, ignore_index=True)
    df_stars_all.to_csv(os.path.join(folder_output, f'{subject}_steps.csv'), index=False)
            
 


# df_stars_all = pd.concat(df_stars_all, ignore_index=True)
# df_stars_all.to_csv(os.path.join(folder_output, 'gaze_steps.csv'), index=False)

# df_activ_all = pd.concat(df_activ_all, ignore_index=True)
# df_activ_all.to_csv(os.path.join(folder_output, 'gaze_activ.csv'), index=False)