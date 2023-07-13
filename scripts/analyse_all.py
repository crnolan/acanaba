# %% [markdown]
# # Basic analysis
#
# NOTE: This requires a python virtualenv already installed with the appropriate dependencies, by e.g.:
#
# `conda create -n acanaba python=3.10 numpy pandas ipykernel jupyter holoviews bokeh datashader seaborn pytables`
#
# Let's load our tracked data and calculate some basic metrics for the pre-reversal and first reversal session.
#
# First we need to import the `pandas` module which allows us to manipulate tabular data. The module provides two basic data structures, DataFrames (table-like) and Series (row-like).

# %%
import pandas as pd
idx = pd.IndexSlice
import numpy as np
import spatial

import holoviews as hv
from holoviews import opts
import datashader as ds
hv.extension('bokeh')
import panel
panel.extension(comms='vscode')

# %% [markdown]
# Let's load our analysed position data.

# %%
from pathlib import Path
from collections import namedtuple
import re
Recording = namedtuple(
    "Recording", ["subject", "session", "task", "acq", "med_path", "dlc_path"])
pattern = (f'sub-*/ses-*/sub-*_ses-*_task-*_acq-*_events.csv')
regex_pattern = (r"sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_events.csv")
data_files = list(Path('../data/operant2023').glob(str(pattern)))
extracted_data = []
for file_path in data_files:
    match = re.search(regex_pattern, str(file_path.name))
    if match:
        sub, ses, task, acq = match.groups()
        dlc_path = re.sub(
            r'acq-.*_events\.csv',
            'vidDLC_resnet50_dlc_acan_masterOct23shuffle1_800000.h5',
            str(file_path))
        data_file = Recording(sub, ses, task, acq, file_path, dlc_path)
        extracted_data.append(data_file)
recordings = pd.DataFrame(extracted_data)
recordings = recordings.set_index(['subject', 'session', 'task', 'acq'])
recordings


# %% [markdown]
# Let's make a function to load a tracking session using the filename (embedded in one row of the above table), so we can load all our data together.

# %%
def load_track_session(session: pd.Series) -> pd.DataFrame:
    '''Load one session analysed by DeepLabCut into a DataFrame'''
    try:
        df = pd.read_hdf(session.iloc[0])
    except FileNotFoundError:
        print(f'File not found: {session.iloc[0]}')
        return None
    df.columns = df.columns.droplevel(0)
    df.index.name = 'frame_id'
    return df

# %% [markdown]
# Now load all the tracking data in one big DataFrame.

# %%
track_df = recordings.dlc_path.groupby(['subject', 'session', 'task', 'acq']).apply(load_track_session)
track_df

# %% [markdown]
# ## Load the session timings.
#
# Each day we ran two MedPC sessions, one for each lever, however we only recorded a single video that spanned both sessions. We somehow need to map the times of the MedPC data to the correct video frames. For this purpose, we had MedPC control an LED positioned in the frame of each video and record the onset and offset times for every LED flash.
#
# We can filter all our positional data by the first and last LED flash in each acquisition run. These values are stored in JSON files associated with each MedPC session.

# %%
from pathlib import Path
import json

# %%
def load_sidecar(filename: str) -> pd.Series:
    with open(filename) as sidecar:
        info = json.load(sidecar)
    return pd.Series(info)

# %%
sidecar_fns = Path('../data/operant2023').glob('*/*/*.json')
info_df = pd.DataFrame([load_sidecar(fn) for fn in sidecar_fns]).set_index(['sub', 'ses', 'acq'])
info_df

# %% [markdown]
# Now we can label the tracking data with the session and acquisition information.

# %%
def get_acq_session(info):
    df = track_df.loc[idx[info.sub, info.ses, :, info.acq, info.firston:info.laston], :]
    return df

# %%
acq_df = pd.concat([get_acq_session(info) for info in info_df.reset_index().itertuples()])
acq_df

# %% [markdown]
# What about the centre point of the head? We need the mean point between the two ears, filtered for only those points where the likelihoods of both points are above a threshold.

# %%
head_centre_mask = (acq_df.stack().loc[idx[:, :, :, :, :, ['likelihood']], idx['leftEar', 'rightEar']] > 0.95).all(axis=1).to_numpy()
head_centre_df = acq_df.stack().loc[idx[:, :, :, :, :, ['x', 'y']], ['leftEar', 'rightEar']].mean(axis=1).unstack()
head_centre_df = head_centre_df.loc[head_centre_mask, :]
head_centre_df['t'] = head_centre_df.index.get_level_values('frame_id') / 30
head_centre_df

# %%
paths = {(sub, ses, acq): hv.Path(head_centre_df.loc[idx[sub, ses, :, acq, :], :])
         for i, sub, ses, acq in info_df.reset_index().loc[:, ['sub', 'ses', 'acq']].itertuples()}
hv.HoloMap(paths, kdims=['sub', 'ses', 'acq']).layout(['acq', 'ses']).cols(2)

# %% [markdown]
# We're plotting trajectories, but what about dwell times? Occupancy maps can give us dwell time.

# %%
bins = np.array((40, 40))
minmax = [head_centre_df.loc[:, ['x', 'y']].min(),
          head_centre_df.loc[:, ['x', 'y']].max()]
path_range = [(a, b) for a, b in zip(*minmax)]

def get_occupancy(df):
    occ = spatial.occupancy_map(df.to_numpy(),
                                bins=bins, smooth=1, max_dt=0.2,
                                range=path_range)
    return hv.Image(occ.hist.T).opts(cmap='viridis', frame_height=200,
                                     data_aspect=(bins[1]/bins[0]))
occ = {(sub, ses, acq): get_occupancy(head_centre_df.loc[idx[sub, ses, :, acq, :], ['t', 'x', 'y']])
       for i, sub, ses, acq in info_df.reset_index().loc[:, ['sub', 'ses', 'acq']].itertuples()}
hv.HoloMap(occ, kdims=['sub', 'ses', 'acq']).layout(['acq', 'ses']).cols(2)

# %% [markdown]
# ## Events analysis
#
# Let's now load up the MedPC events and look at the event-related action

# %%
def load_med_session(session: pd.Series) -> pd.DataFrame:
    '''Load one session analysed by DeepLabCut into a DataFrame'''
    try:
        df = pd.read_csv(session.iloc[0])
    except FileNotFoundError:
        print(f'File not found: {session.iloc[0]}')
        return None
    df.index.name = 'event_id'
    return df.loc[:, ['onset', 'value']]

events = recordings.med_path.groupby(['subject', 'session', 'task', 'acq']).apply(load_med_session)
events

# %%
