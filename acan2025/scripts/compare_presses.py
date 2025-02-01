# %%
import json
from pathlib import Path
from collections import namedtuple
import re
import numpy as np
import pandas as pd
from utils import load_recordings, load_track_session
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
idx = pd.IndexSlice

# %%
data_path = Path('../rawdata')
rec_cols = ['subject', 'session', 'task', 'acq']

# %%
recordings = load_recordings(data_path)
recordings

# %% [markdown]
#
# Now load all the tracking data in one big DataFrame.

# %%
track_df = recordings[['dlc_path', 'firston', 'laston', 'firstmed']].groupby(rec_cols).apply(
    lambda x: load_track_session(*(x.iloc[0])))
track_df

# %%
# recordings.eval('(laston - firston) / (lastmed - firstmed)')

# %% [markdown]
#
# These should all be close to zero if the tracking data is correct.

# %%
recordings.eval('30 * (lastmed - firstmed) + firston - laston')

# %%
# _df = track_df.stack('bodyparts', future_stack=True)
# _df = _df.mask(_df['likelihood'] < 0.5).drop('likelihood', axis=1)
# _df = _df.unstack('bodyparts').reorder_levels(['bodyparts', 'coords'], axis=1)
# _df = _df.sort_index(axis=1)
# _df.iloc[30:100]

# %% [markdown]
#
# We'll want to use both position and orientation data, so create a
# function to calculate the latter.

# %%
def calc_orientation(left: pd.DataFrame, right: pd.DataFrame,
                     name: str) -> pd.DataFrame:
    offset = left - right
    theta = np.arctan2(offset.y, offset.x) + np.pi / 2
    _df = pd.DataFrame({
        'theta': theta
        # 'likelihood': np.minimum(left['likelihood'], right['likelihood'])
    })
    _df.columns = pd.MultiIndex.from_product([[name], _df.columns])
    _df.index = left.index
    _df.columns.names = ['bodyparts', 'coords']
    return _df


# %%
head_theta = track_df.groupby(rec_cols, group_keys=False).apply(
    lambda x: calc_orientation(x['left_ear'], x['right_ear'], 'head'))
track_df = pd.concat([track_df, head_theta], axis=1)

# %%
track_df

# %% [markdown]
#
# Now let's calculate speed and angular speed.

# %%
def calc_speed(df: pd.DataFrame) -> pd.DataFrame:
    def unwrap_diff(x):
        _s = pd.Series(np.nan, index=x.index)
        _s.loc[~x.isna()] = np.unwrap(x.loc[~x.isna()])
        return _s.diff()
    dtheta = df.loc[:, idx[:, ['theta']]].apply(unwrap_diff)
    dtheta.rename(columns={'theta': 'dtheta'}, inplace=True)
    dxy = df.loc[:, idx[:, ['x', 'y']]].diff()
    speed = dxy.pow(2).stack('bodyparts', future_stack=True).sum(skipna=False, axis=1).pow(0.5).unstack('bodyparts')
    speed.columns = pd.MultiIndex.from_product([speed.columns, ['speed']])
    return pd.concat([speed, dtheta], axis=1).sort_index(axis=1)


speed = track_df.groupby(rec_cols, group_keys=False).apply(calc_speed)


# %% [markdown]
#
# Let's get windows around the

# %%
# def get_event_windows(df, window_range=(-0.1, 10)):
#     x = df.iloc[0]
#     p0, p1 = window_range
#     ev_window = head_centre_masked_df.loc[idx[x.subject, x.session, x.task, x.acq, :, (x.onset+p0):(x.onset+p1)]]
#     ev_window['window_offset'] = np.arange(len(ev_window))
#     ev_window.set_index('window_offset', append=True, inplace=True)
#     return ev_window.reset_index(['subject', 'session', 'task', 'acq'], drop=True)
