# %% [markdown]
#
# # Quantifying movement in a single subject
#
# Here we take a look at movement data from a single subject before and
# after reversal.

# %% [markdown]
#
#

# %%
import json
from pathlib import Path
import pandas as pd
import numpy as np

# %%
sub_path = Path('../rawdata/sub-brain')

# %% [markdown]
#
# Let's first load up some metadata about one acquisition.

# %%
json_path = (sub_path / 'ses-RR20.03/'
             / 'sub-brain_ses-RR20.03_task-RR20_acq-A.json')
with open(json_path) as f:
    recording = json.load(f)
recording

# %% [markdown]
#
# firston and laston are the times of the first and last frames


# %% [markdown]
#
# We can read in a DeepLabCut tracking file (HDF5 format) using the
# pandas read_hdf function.

# %%
dlc_path = (sub_path / 'ses-RR20.03/'
            / 'sub-brain_ses-RR20.03_task-RR20_acq-A_vidcroppedDLC_HrnetW32_acanJan28shuffle5_detector_360_snapshot_110_filtered.h5')
tracking = pd.read_hdf(dlc_path)
tracking

# %% [markdown]
#
# It initially looks like there is no data, but that is the very start
# and end of the tracking session. Let's look in the middle.
#
# `len(tracking)` gives the number of rows in the table, and the `iloc`
# method allows us to access rows by index. We can use the notation
# `start:end` to request a range of rows.

# %%
# This will give us the middle(ish) 10 rows
tracking.iloc[(len(tracking) // 2):((len(tracking) // 2) + 10)]

# %% [markdown]
#
# There is a strange "scorer" level in the columns, we don't need that
# so let's drop it.

# %%
tracking.columns = tracking.columns.droplevel(0)
tracking

# %% [markdown]
#
# The index currently doesn't have an identifier, let's call it "frame_id".

# %%
tracking.index.name = 'frame_id'
tracking


# %% [markdown]
#
# This data has already been filtered based on the likelihood column
# using a median filter. We can just drop the likelihood column. At the
# same time, we can filter out all the frames outside the first and last
# LED flashes.

# %%
tracking = tracking.loc[recording['firston']:recording['laston'],
                        (slice(None), ['x', 'y'])].sort_index(axis=1)
tracking

# %% [markdown]
#
# DLC uses -1 to indicate missing data in this case, which we can
# replace with NaN (not a number).

# %%
tracking = tracking.mask(tracking < 0)

# %% [markdown]
#
# Let's also get the time in seconds from the start of the session.

# %%
tracking['time'] = ((tracking.index.get_level_values('frame_id')
                     - recording['firston'])
                    / 30 + recording['firstmed'])
tracking.set_index('time', append=True, inplace=True)
tracking

# %% [markdown]
#
# At this point we can plot the animal's position over time.

# %%
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import panel
panel.extension(comms='vscode')

# %%
neck = tracking.loc[:, ('neck', ('x', 'y'))].droplevel(0, axis=1)
hv.Path(neck).opts(opts.Path(height=400, width=400, invert_yaxis=True))

# %% [markdown]
#
# That doens't really tell us much about the behaviour. Let's try to
# specifically look at the time around the lever presses.
#
# First get a function to load the events sessions (this is basically
# just doing what we did in the first script)

# %%
def load_events_session(events_path, block):
    '''Load one session analysed by DeepLabCut into a DataFrame'''
    try:
        df = pd.read_csv(events_path)
    except FileNotFoundError:
        print(f'File not found: {events_path}')
        return None
    df = df.replace({'lp': 'llp' if block[0] == 'L' else 'rlp'})
    df['onset'] = pd.to_timedelta(df['onset'], unit='s')
    return df.fillna(0.01).set_index('onset')


# %%
events_path = (sub_path / 'ses-RR20.03/'
               / 'sub-brain_ses-RR20.03_task-RR20_acq-A_events.csv')
events = load_events_session(events_path, recording['block'])
events


# %% [markdown]
#
# We need a function to extract the data around the event times.

# %%
def get_event_windows(df, onset, window_range=(-0.5, 0.0)):
    nearest_i = np.argmin(np.abs(df.index.get_level_values('time') - onset))
    nearest_frame = df.index.get_level_values('frame_id')[nearest_i]
    frame_range = np.rint(np.array(window_range) * 30).astype(int)
    fs, fe = frame_range + nearest_frame
    ev_window = df.loc[(slice(fs, fe), slice(None)), :].copy()
    ev_window['window_offset'] = np.arange(frame_range[0], frame_range[1]+1,
                                           dtype=int)
    return ev_window.set_index('window_offset')


# %%
event_cols = ['onset', 'event_id']
lp_events_df = events.query('event_id.isin(["llp", "rlp"])').reset_index()
lp_events_df['onset'] = lp_events_df['onset'].dt.total_seconds()
lp_windows_df = lp_events_df.groupby(event_cols)[event_cols].apply(
    lambda x: get_event_windows(neck,
                                x.iloc[0].onset))
lp_beta = lp_windows_df.unstack('window_offset')

# %%
# Let's plot the data around the lever presses
hv.Path(list(lp_windows_df.groupby(['onset']).apply(lambda x: x.to_numpy())))

# %%
