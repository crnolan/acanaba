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
# Remy / Danger / Pinky / The Brain
# %%
import pandas as pd
idx = pd.IndexSlice
import numpy as np
import spatial
from behapy.events import find_events

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
# data_path = Path('../data/operant2023')
data_path = Path(r'C:\Users\cnolan\UNSW\ACAN - ABA - ABA\ACAN2023\operant\organised')
Recording = namedtuple(
    "Recording", ["subject", "session", "task", "acq", "med_path", "dlc_path"])
pattern = (f'sub-*/ses-*/sub-*_ses-*_task-*_acq-*_events.csv')
regex_pattern = (r"sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_events.csv")
# data_files = list(Path('../data/operant2023').glob(str(pattern)))
data_files = list(data_path.glob(str(pattern)))
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
sidecar_fns = data_path.glob('*/*/*.json')
info_df = pd.DataFrame([load_sidecar(fn) for fn in sidecar_fns]).set_index(['sub', 'ses', 'acq'])
# Check the frame rate is consistent
info_df['fps'] = info_df.eval('(laston - firston) / (lastmed - firstmed)')
info_df['fps30_lastframe'] = info_df.eval('30 * (lastmed - firstmed) + firston')
info_df['frame_diff'] = info_df.eval('fps30_lastframe - laston')
info_df

# %% [markdown]
# Now we can label the tracking data with the session and acquisition information.

# %%
def get_acq_session(info):
    df = track_df.loc[idx[info.sub, info.ses, :, info.acq, info.firston:info.laston], :]
    df
    return df

# %%
acq_df = pd.concat([get_acq_session(info) for info in info_df.reset_index().itertuples()])
# TODO: Convert frame number to MedPC time
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
info_df

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
events = events.reset_index().drop('event_id', axis=1).rename({'value': 'event_id'}, axis=1).set_index(['subject', 'session', 'task', 'acq', 'onset'])
events['duration'] = 0.01
events = events[['duration', 'event_id']]
events


# %%
def _concat_events(events,
                   new_event,
                   new_event_id):
    new_events = pd.concat([new_event],
                           keys=[new_event_id],
                           names=['event_id'])
    new_events = new_events.reset_index('event_id').loc[:, ['duration', 'event_id']]
    return pd.concat([events, new_events]).sort_index()


def _find_and_concat_events(events,
                            new_event_id,
                            reference,
                            source,
                            reject=[],
                            direction='forward',
                            allow_exact_matches=True):
    sub_events = find_events(events, reference, source, reject, direction,
                             allow_exact_matches)
    return _concat_events(events, sub_events, new_event_id)


def _get_nonevent(events, sub_events):
    nonevent = events.loc[:, ['duration']].merge(
        sub_events.loc[:, []],
        how='left', left_index=True, right_index=True, indicator=True)
    return nonevent.loc[nonevent._merge == 'left_only', ['duration']]


def _get_nonevent_and_concat(events, new_event_id, match_event, sub_events):
    if isinstance(sub_events, str):
        sub_events = [sub_events]
    nonevent = _get_nonevent(events.loc[events.event_id == match_event, :],
                             events.loc[events.event_id.isin(sub_events), :])
    return _concat_events(events, nonevent, new_event_id)


# %% [markdown]
#
# Firstly get activity around all rewarded mag events
events = _find_and_concat_events(events, 'REWmag', 'Mag', 'Rew', direction='forward')


# %%
from shapely.geometry import Polygon
from shapely.vectorized import contains

lp_df = pd.read_csv('region_coordinates.csv', index_col=['subject'])

# %%
def in_region_group(group):
    regions = lp_df.loc[group.index[0][0]]
    session = group.index[0][1]
    left_poly = Polygon([(regions.ll_tl_x, regions.ll_tl_y),
                         (regions.ll_br_x, regions.ll_tl_y),
                         (regions.ll_br_x, regions.ll_br_y),
                         (regions.ll_tl_x, regions.ll_br_y)])
    right_poly = Polygon([(regions.rl_tl_x, regions.rl_tl_y),
                          (regions.rl_br_x, regions.rl_tl_y),
                          (regions.rl_br_x, regions.rl_br_y),
                          (regions.rl_tl_x, regions.rl_br_y)])
    if session == 'prerev':
        if regions.ll_pre == 'PUR':
            pur_poly = left_poly
            grn_poly = right_poly
        else:
            pur_poly = right_poly
            grn_poly = left_poly
    else:
        if regions.ll_post == 'PUR':
            pur_poly = left_poly
            grn_poly = right_poly
        else:
            pur_poly = right_poly
            grn_poly = left_poly
    zones = {
        'PUR': pur_poly,
        'GRN': grn_poly
    }
    df = pd.DataFrame({
        zone: contains(poly, group.loc[:, 'x'].to_numpy(), group.loc[:, 'y'].to_numpy()) for zone, poly in zones.items()
    })
    return df.set_index(group.index)

region_occ = head_centre_df.groupby(['subject', 'session'], group_keys=False).apply(in_region_group)
region_occ.columns.name = 'region'
region_occ


# %%
region_occ_filtered = region_occ.loc[region_occ.any(axis=1), :]
region_occ_filtered = region_occ_filtered.stack()
region_occ_filtered = region_occ_filtered[region_occ_filtered].to_frame().reset_index().set_index(['subject', 'session', 'task', 'acq', 'frame_id']).drop(0, axis=1)
region_occ_filtered

# %%
region_occ_filtered['group'] = region_occ_filtered.groupby(['subject', 'session', 'task', 'acq'], as_index=False).transform(lambda x: (x != x.shift(1)).cumsum())
region_occ_filtered

# %%
bodyparts_in_region_groups = region_occ_filtered.reset_index().groupby(['subject', 'session', 'task', 'acq', 'group', 'region']).agg(lambda x: x.iloc[-1] - x.iloc[0])
bodyparts_in_region_occ = bodyparts_in_region_groups.groupby(['subject', 'session', 'task', 'acq', 'region']).sum()
bodyparts_in_region_occ

# %%
bodyparts_in_region_occ['time'] = bodyparts_in_region_occ['frame_id'] / 30


# %%
import seaborn as sns
sns.catplot(data=bodyparts_in_region_occ.reset_index(),
            x='region', y='time', hue='acq', row='session',
            kind='point', scale=0.4)

# %%
p1 = sns.pointplot(data=bodyparts_in_region_occ.time.loc[idx[:, 'prerev', :, :, :, :]].reset_index(), x='region', y='time', units='subject', hue='subject', scale=0.4)
p1.legend().remove()

# %%
p2 = sns.pointplot(data=bodyparts_in_region_occ.time.loc[idx[:, 'rev.01', :, :, :, :]].reset_index(), x='region', y='time', units='subject', hue='subject', scale=0.4)
p2.legend().remove()


# %%
bodyparts_in_region_occ

# %%
def dd(df):
    xy = df.diff()
    return xy.pow(2).sum(axis=1).pow(0.5)
hc_dd = head_centre_df.groupby(['subject', 'session', 'task', 'acq'], group_keys=False).apply(dd)
hc_dist_travelled= hc_dd.groupby(['subject', 'session', 'task', 'acq']).sum()

# %%
hc_dist_travelled.name = 'distance_travelled'
p1 = sns.pointplot(data=hc_dist_travelled.reset_index(), x='session', y='distance_travelled', units='subject', hue='subject', scale=0.4)
p1.legend().remove()

# %%
