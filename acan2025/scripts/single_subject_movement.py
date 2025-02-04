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

# %%
sub_path = Path('../rawdata/sub-brain')

# %% [markdown]
#
# Let's load up the
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
# using a median filter. We can just drop the likelihood column. DLC
# uses -1 to indicate missing data in this case, which we can replace
# with NaN (not a number).

# %%
tracking = tracking.loc

# %%
def load_track_session(dlc_path, firston, laston, firstmed,
                       prefiltered=True) -> pd.DataFrame:
    '''Load one session analysed by DeepLabCut into a DataFrame'''
    try:
        df = pd.read_hdf(dlc_path)
    except FileNotFoundError:
        print(f'File not found: {dlc_path}')
        return None
    df.columns = df.columns.droplevel(0)
    df.index.name = 'frame_id'
    df = df.loc[firston:laston, (slice(None), ['x', 'y'])].sort_index(axis=1)
    if prefiltered:
        # Drop the likelihood column and substitute NaN for -1
        df = df.mask(df < 0)
    df['time'] = (df.index.get_level_values('frame_id') - firston) / 30 + firstmed
    df.set_index('time', append=True, inplace=True)
    return df


# %%
events_paths = [str(p) for p in data_path.glob('*_events.csv')]
events_paths
