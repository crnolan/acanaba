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
import json
from pathlib import Path
from collections import namedtuple
import re
import numpy as np
import pandas as pd
idx = pd.IndexSlice
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
dlc_threshold = 0.5
data_path = Path(r'C:\Users\cnolan\UNSW\ACAN - ABA - ABA\ACAN2023\operant\organised')
Recording = namedtuple(
    "Recording", ["subject", "session", "task", "acq", "med_path", "dlc_path"])
# pattern = (f'sub-*/ses-*/sub-*_ses-*_task-*_acq-*_events.csv')
# regex_pattern = (r"sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_events.csv")
pattern = (f'sub-*/ses-*/sub-*_ses-*_task-*_acq-*_events.csv')
regex_pattern = (r"sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_events.csv")
# data_files = list(Path('../data/operant2023').glob(str(pattern)))
data_files = list(data_path.glob(str(pattern)))
extracted_data = []
for file_path in data_files:
    match = re.search(regex_pattern, str(file_path.name))
    if match:
        sub, ses, task, acq = match.groups()
        if ses in ['prerev', 'rev.01']:
            dlc_postfix = 'vidDLC_resnet50_dlc_acan_masterOct23shuffle1_800000.h5'
        else:
            dlc_postfix = 'vidDLC_Resnet50_dlc_acan_masterOct23shuffle2_snapshot_060_filtered.h5'
        dlc_path = re.sub(
            r'acq-.*\.json',
            dlc_postfix,
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
    df = track_df.loc[idx[info.sub, info.ses, :, info.acq, info.firston:info.laston], :].copy()
    _points = ['leftEar', 'rightEar']
    _df = pd.DataFrame({
        'x': df.loc[:, idx[_points, 'x']].mean(axis=1),
        'y': df.loc[:, idx[_points, 'y']].mean(axis=1),
        'likelihood': df.loc[:, idx[_points, 'likelihood']].min(axis=1)
    })
    _df.columns = pd.MultiIndex.from_product([['head'], _df.columns])
    _df.columns.names = ['bodyparts', 'coords']
    df = pd.concat([df, _df], axis=1)
    frame_ids = df.index.get_level_values('frame_id')
    # df['time'] = (frame_ids - frame_ids.min()) / info.fps + 0.7
    df['time'] = (frame_ids - frame_ids.min()) / 30 + 0.7
    return df.set_index('time', append=True)

# %%
acq_df = pd.concat([get_acq_session(info) for info in info_df.reset_index().itertuples()])
acq_df

# %%
def calc_speed(df, group_columns):
    _df = df.copy()
    _xydiff = _df.loc[:, ['x', 'y']].groupby(group_columns).diff()
    _df['speed'] = _xydiff.pow(2).sum(skipna=False, axis=1).pow(0.5)
    # Filter by likelihood; drop all rows and row-1s (because speed is
    # calculated between rows)
    _df.loc[_df.loc[:, 'likelihood'] < dlc_threshold, ['speed']] = pd.NA
    _ls = _df.loc[:, 'likelihood'].groupby(group_columns).shift(1)
    _df.loc[_ls < dlc_threshold, ['speed']] = pd.NA

    # Now get a rolling mean of speed
    # _df['mean_speed'] = _df['speed'].groupby(group_columns).apply(
    #     lambda x: print(x.rolling(15, center=True).mean().head(20)))
    _df['mean_speed'] = _df['speed'].groupby(group_columns, group_keys=False).apply(
        lambda x: x.rolling(15, center=True).mean())
    return _df


acq_df = calc_speed(acq_df.stack('bodyparts'), ['subject', 'session', 'task', 'acq', 'bodyparts'])
acq_df = acq_df.unstack('bodyparts').reorder_levels([1, 0], axis=1).sort_index(axis=1)

# %% [markdown]
# What about the centre point of the head? We need the mean point between the two ears, filtered for only those points where the likelihoods of both points are above a threshold.

# %%
# head_centre_mask = (acq_df.stack().loc[idx[:, :, :, :, :, :, ['likelihood']], idx['leftEar', 'rightEar']] > 0.95).all(axis=1).to_numpy()
# head_centre_df = acq_df.stack().loc[idx[:, :, :, :, :, :, ['x', 'y']], ['leftEar', 'rightEar']].mean(axis=1).unstack()
# head_centre_df = head_centre_df.loc[head_centre_mask, :]
# # head_centre_df['t'] = head_centre_df.index.get_level_values('frame_id') / 30
# head_centre_df
head_centre_df = acq_df.loc[acq_df.loc[:, idx['head', 'likelihood']] > dlc_threshold, idx['head', ['x', 'y']]]
head_centre_df = head_centre_df.droplevel(0, axis=1)
head_centre_masked_df = acq_df.loc[: , ('head')].copy()
head_centre_masked_df['mask'] = head_centre_masked_df.loc[:, idx['likelihood']] > dlc_threshold
head_centre_masked_df = head_centre_masked_df.drop('likelihood', axis=1)

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
occ = {(sub, ses, acq): get_occupancy(head_centre_df.reset_index(['time']).loc[idx[sub, ses, :, acq, :], ['time', 'x', 'y']])
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

# %%
events = _find_and_concat_events(events, 'REWmag', 'Mag', 'Rew', direction='forward')
events = _find_and_concat_events(events, 'REWMAGlp', 'LP', 'REWmag', direction='forward')

# %%
events

# %% [markdown]
#
# Let's take a look specifically at the rewarded magazine entries and
# the time right after them.

# %%
def get_event_windows(df, window_range=(-0.1, 10)):
    x = df.iloc[0]
    p0, p1 = window_range
    ev_window = head_centre_masked_df.loc[idx[x.subject, x.session, x.task, x.acq, :, (x.onset+p0):(x.onset+p1)]]
    ev_window['window_offset'] = np.arange(len(ev_window))
    ev_window.set_index('window_offset', append=True, inplace=True)
    return ev_window.reset_index(['subject', 'session', 'task', 'acq'], drop=True)


onset_groupby = ['subject', 'session', 'task', 'acq', 'onset', 'event_id']
REWmag_groups = events.loc[events.event_id == 'REWmag'].reset_index().groupby(onset_groupby)
REWmag_post = REWmag_groups[['subject', 'session', 'task', 'acq', 'onset']].apply(get_event_windows, window_range=(-0.1, 5))
rew_groups = events.loc[events.event_id == 'Rew'].reset_index().groupby(onset_groupby)
rew_post = rew_groups[['subject', 'session', 'task', 'acq', 'onset']].apply(get_event_windows, window_range=(-0.1, 1))
# REWmag_windows.droplevel(['frame_id', 'time'], axis=0).unstack('window_offset')
REWMAGlp_groups = events.loc[events.event_id == 'REWMAGlp'].reset_index().groupby(onset_groupby)
REWMAGlp_post = REWMAGlp_groups[['subject', 'session', 'task', 'acq', 'onset']].apply(get_event_windows, window_range=(-2, 0))
lp_groups = events.loc[events.event_id.isin(['LLP', 'RLP'])].reset_index().groupby(onset_groupby)
lp_post = lp_groups[['subject', 'session', 'task', 'acq', 'onset']].apply(get_event_windows, window_range=(-2, 0))

# %%
# Interpolate invalid values
REWmag_post.loc[REWmag_post['mask'] == False, ['x', 'y']] = np.nan
REWmag_post.loc[:, ['x', 'y']] = REWmag_post.loc[:, ['x', 'y']].interpolate()
rew_post.loc[rew_post['mask'] == False, ['x', 'y']] = np.nan
rew_post.loc[:, ['x', 'y']] = rew_post.loc[:, ['x', 'y']].interpolate()
REWMAGlp_post.loc[REWMAGlp_post['mask'] == False, ['x', 'y']] = np.nan
REWMAGlp_post.loc[:, ['x', 'y']] = REWMAGlp_post.loc[:, ['x', 'y']].interpolate()
lp_post.loc[lp_post['mask'] == False, ['x', 'y']] = np.nan
lp_post.loc[:, ['x', 'y']] = lp_post.loc[:, ['x', 'y']].interpolate()

# %%
# Get a unique id for each onset within the session
REWmag_post = REWmag_post.join(
    REWmag_post.index.droplevel(['frame_id', 'time', 'window_offset']).unique().sort_values().to_frame().loc[:, []].groupby(['subject', 'session', 'task', 'acq']).cumcount().rename('onset_id'),
    how='left').set_index('onset_id', append=True).reorder_levels(['subject', 'session', 'task', 'acq', 'onset_id', 'onset', 'event_id', 'frame_id', 'time', 'window_offset'])
rew_post = rew_post.join(
    rew_post.index.droplevel(['frame_id', 'time', 'window_offset']).unique().sort_values().to_frame().loc[:, []].groupby(['subject', 'session', 'task', 'acq']).cumcount().rename('onset_id'),
    how='left').set_index('onset_id', append=True).reorder_levels(['subject', 'session', 'task', 'acq', 'onset_id', 'onset', 'event_id', 'frame_id', 'time', 'window_offset'])
REWMAGlp_post = REWMAGlp_post.join(
    REWMAGlp_post.index.droplevel(['frame_id', 'time', 'window_offset']).unique().sort_values().to_frame().loc[:, []].groupby(['subject', 'session', 'task', 'acq']).cumcount().rename('onset_id'),
    how='left').set_index('onset_id', append=True).reorder_levels(['subject', 'session', 'task', 'acq', 'onset_id', 'onset', 'event_id', 'frame_id', 'time', 'window_offset'])
lp_post = lp_post.join(
    lp_post.index.droplevel(['frame_id', 'time', 'window_offset']).unique().sort_values().to_frame().loc[:, []].groupby(['subject', 'session', 'task', 'acq']).cumcount().rename('onset_id'),
    how='left').set_index('onset_id', append=True).reorder_levels(['subject', 'session', 'task', 'acq', 'onset_id', 'onset', 'event_id', 'frame_id', 'time', 'window_offset'])

# %%
paths = {(sub, ses): hv.Overlay(
            [hv.Path(list(REWmag_post.loc[idx[sub, ses, :, acq, :, :, :, :, :, :], ['x', 'y']].groupby(['onset']).apply(lambda x: x.to_numpy())))
             for acq in ['A', 'B']])
         for i, sub, ses in info_df.loc[idx[:, ['prerev', 'rev.01'], :], :].reset_index().loc[:, ['sub', 'ses']].itertuples()}
hv.HoloMap(paths, kdims=['sub', 'ses']).layout(['ses']).cols(2).opts(opts.Path(frame_width=400, frame_height=400, alpha=0.5))

# %%
# How about just the first 5 onsets?
paths = {(sub, ses): hv.Overlay(
            [hv.Path(list(REWmag_post.loc[idx[sub, ses, :, acq, :5, :, :, :, :, :], ['x', 'y']].groupby(['onset']).apply(lambda x: x.to_numpy())))
             for acq in ['A', 'B']])
         for i, sub, ses in info_df.loc[idx[:, ['prerev', 'rev.01'], :], :].reset_index().loc[:, ['sub', 'ses']].itertuples()}
hv.HoloMap(paths, kdims=['sub', 'ses']).layout(['ses']).cols(2).opts(opts.Path(frame_width=400, frame_height=400, alpha=0.5))

# %%
# Lever presses?
paths = {(sub, ses): hv.Overlay(
            [hv.Path(list(REWMAGlp_post.loc[idx[sub, ses, :, acq, :, :, :, :, :, :], ['x', 'y']].groupby(['onset']).apply(lambda x: x.to_numpy())))
             for acq in ['A', 'B']])
         for i, sub, ses in info_df.loc[idx[:, ['prerev', 'rev.01'], :], :].reset_index().loc[:, ['sub', 'ses']].itertuples()}
hv.HoloMap(paths, kdims=['sub', 'ses']).layout(['ses']).cols(2).opts(opts.Path(frame_width=400, frame_height=400, alpha=0.5))

# %%
# How about just the first 5 lever presses?
paths = {(sub, ses): hv.Overlay(
            [hv.Path(list(REWMAGlp_post.loc[idx[sub, ses, :, acq, :5, :, :, :, :, :], ['x', 'y']].groupby(['onset']).apply(lambda x: x.to_numpy())))
             for acq in ['A', 'B']])
         for i, sub, ses in info_df.loc[idx[:, ['prerev', 'rev.01'], :], :].reset_index().loc[:, ['sub', 'ses']].itertuples()}
hv.HoloMap(paths, kdims=['sub', 'ses']).layout(['ses']).cols(2).opts(opts.Path(frame_width=400, frame_height=400, alpha=0.5))

# %%
# Lever presses during deval?
paths = {(sub, ses): hv.Overlay(
            [hv.Path(list(lp_post.loc[idx[sub, ses, :, :, :, :, event, :, :, :], ['x', 'y']].groupby(['onset']).apply(lambda x: x.to_numpy())))
             for event in ['LLP', 'RLP']])
         for i, sub, ses in info_df.loc[idx[:, ['deval-rev.01', 'deval-rev.02'], :], :].reset_index().loc[:, ['sub', 'ses']].itertuples()}
hv.HoloMap(paths, kdims=['sub', 'ses']).layout(['ses']).cols(2).opts(opts.Path(frame_width=400, frame_height=400, alpha=0.5))

# %% [markdown]
#
# What about the approach velocity to the magazine after reward?

# %%
import panel as pn
import hvplot.pandas
rew_post.loc[:, ['speed']].groupby(onset_groupby).mean().reset_index().hvplot.violin(
    y='speed', by=['subject', 'session', 'acq'], color='acq', width=1200, height=400)

# %%
rew_post.loc[:, ['speed']].groupby(['subject', 'session', 'task', 'acq', 'onset_id']).mean().reset_index().hvplot.line(
    x='onset_id', y='speed', by=['session', 'acq'], groupby='subject', width=1200, height=400)


# %% [markdown]
#

# %%
REWmag_windows['speed'] = REWmag_windows.loc[:, ['x', 'y']].groupby(onset_groupby).apply(lambda x: print(x.diff().pow(2).sum(axis=1).pow(0.5) / 30))
REWmag_windows

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
