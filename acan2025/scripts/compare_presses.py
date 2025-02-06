# %%
import json
from pathlib import Path
from collections import namedtuple
import re
import numpy as np
import pandas as pd
from utils import load_recordings, load_track_session, load_events_session
import logging
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

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

# %% [markdown]
#
# And the same for events.

# %%
events_df = recordings[['events_path', 'block']].groupby(rec_cols).apply(
    lambda x: load_events_session(*(x.iloc[0])))

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
track_df

# %%
track_filt = track_df.loc[:, idx[['head', 'neck'], :]]


# %% [markdown]
#
# Now let's calculate speed and angular speed.

# %%
def calc_speed(df: pd.DataFrame) -> pd.DataFrame:
    def unwrap_diff(x):
        _s = pd.Series(np.nan, index=x.index)
        _s.loc[~x.isna()] = np.unwrap(x.loc[~x.isna()])
        return _s.diff().abs()
    dtheta = df.loc[:, idx[:, ['theta']]].apply(unwrap_diff)
    dtheta.rename(columns={'theta': 'dtheta'}, inplace=True)
    dxy = df.loc[:, idx[:, ['x', 'y']]].diff()
    speed = dxy.pow(2).stack('bodyparts', future_stack=True).sum(skipna=False, axis=1).pow(0.5).unstack('bodyparts')
    speed.columns = pd.MultiIndex.from_product([speed.columns, ['speed']])
    return pd.concat([speed, dtheta], axis=1).sort_index(axis=1)


speed = track_df.groupby(rec_cols, group_keys=False).apply(calc_speed)
speed_filt = speed.loc[:, idx[['head', 'neck'], :]]


# %% [markdown]
#
# We need data around our event times

# %%
def get_event_windows(df, onset, window_range=(-0.5, 0.0)):
    nearest_i = np.argmin(np.abs(df.index.get_level_values('time') - onset))
    nearest_frame = df.index.get_level_values('frame_id')[nearest_i]
    frame_range = np.rint(np.array(window_range) * 30).astype(int)
    fs, fe = frame_range + nearest_frame
    ev_window = df.loc[(slice(fs, fe), slice(None)), :].copy()
    try:
        ev_window['window_offset'] = np.arange(frame_range[0], frame_range[1]+1,
                                               dtype=int)
    except ValueError:
        logging.warning(f'Dropped event at onset {onset}, '
                        f'likely missing frame data.')
        return None
    return ev_window.set_index('window_offset')


# %%
event_cols = ['subject', 'session', 'task', 'acq', 'onset', 'event_id']
lp_events_df = events_df.query('event_id.isin(["llp", "rlp"])').reset_index()
lp_windows_df = lp_events_df.groupby(event_cols)[event_cols].apply(
    lambda x: get_event_windows(speed_filt.loc[(x.iloc[0].subject,
                                                x.iloc[0].session,
                                                x.iloc[0].task,
                                                x.iloc[0].acq), :],
                                x.iloc[0].onset))
lp_beta = lp_windows_df.unstack('window_offset')


# %%
def fit_logistic_regression(df: pd.DataFrame, cv=40,
                            max_iter=10000) -> pd.DataFrame:
    clf = LogisticRegressionCV(
        cv=cv, max_iter=max_iter, class_weight='balanced', n_jobs=-1).fit(
            df.values, df.index.get_level_values('event_id'))
    score = clf.score(df.values, df.index.get_level_values('event_id'))
    cm = pd.DataFrame(
        confusion_matrix(df.index.get_level_values('event_id'),
                         clf.predict(df.values),
                         labels=['llp', 'rlp']),
        index=pd.Index(['llp', 'rlp'], name='true'),
        columns=pd.Index(['llp', 'rlp'], name='predicted'))
    return pd.Series({'clf': clf,
                      'score': score,
                      'confusion_matrix': cm})

# %%
clfs = lp_beta.interpolate().groupby(['subject', 'session', 'task']).apply(
    fit_logistic_regression)


# %% [markdown]
#
# We have a classifier for each subject and session. What we want to know
# is how well each classifier does on another day's data.

# %%
def test_logistic_regression(clf, df: pd.DataFrame) -> pd.DataFrame:
    score = clf.score(df.values, df.index.get_level_values('event_id'))
    cm = pd.DataFrame(
        confusion_matrix(df.index.get_level_values('event_id'),
                         clf.predict(df.values),
                         labels=['llp', 'rlp']),
        index=pd.Index(['llp', 'rlp'], name='true'),
        columns=pd.Index(['llp', 'rlp'], name='predicted'))

    return pd.Series({'score': score,
                      'confusion_matrix': cm})


def _tlr(df, train, test):
    try:
        return test_logistic_regression(df.loc[(df.name, *train)].clf,
                                        lp_beta.interpolate().loc[(df.name, *test)])
    except KeyError:
        return None

# %%
clfs.groupby(['subject']).apply(lambda x: _tlr(x, ('RR20.03', 'RR20'), ('RR20prerev.01', 'RR20prerev')))

# %%
clfs.groupby(['subject']).apply(lambda x: _tlr(x, ('RR20prerev.01', 'RR20prerev'), ('RR20rev.01', 'RR20rev')))

# %%
