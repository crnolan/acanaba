# %%
from pathlib import Path
import json
import pandas as pd
import re
idx = pd.IndexSlice

# %%
def load_track_session(filename: str) -> pd.DataFrame:
    '''Load one session analysed by DeepLabCut into a DataFrame'''
    df = pd.read_hdf(filename)
    df.columns = df.columns.droplevel(0)
    df.index.name = 'frame_id'
    return df


def load_sidecar(filename: str) -> pd.Series:
    with open(filename) as sidecar:
        info = json.load(sidecar)
    return pd.Series(info)


def get_acq_session(info, track_df):
    df = track_df.loc[idx[info.ses, info.firston:info.laston], :]
    df = df.assign(acq=info.acq).set_index('acq', append=True)
    df.index = df.index.reorder_levels(['session_id', 'acq', 'frame_id'])
    return df


# %%
tracked_template = 'sub-(.*)_ses-(.*)_task.*'
pattern = re.compile(tracked_template)
def parse_tracked_keys(filename):
    matches = pattern.match(filename)
    return matches[1], matches[2]

rawdata_path = Path('../rawdata')
pos_df = pd.concat({parse_tracked_keys(fn.name): load_track_session(fn)
                    for fn in rawdata_path.glob('*/*/*.h5')},
                   names=['subject_id', 'session_id'])

# %%
sidecar_fns = Path('../rawdata').glob('*/*/*.json')
info_df = pd.DataFrame([load_sidecar(fn) for fn in sidecar_fns]).set_index(['sub', 'ses', 'acq'])

# %%
def get_acq_session(info, track_df):
    df = track_df.loc[info.sub, info.ses, :].loc[info.firston:info.laston]
    df = df.assign(subject_id=info.sub, session_id=info.ses, acq_id=info.acq)
    df = df.set_index(['subject_id', 'session_id', 'acq_id'], append=True)
    df.index = df.index.reorder_levels(['subject_id', 'session_id', 'acq_id', 'frame_id'])
    return df

# info_df.reset_index().apply(get_acq_session, axis=1, track_df=pos_df)
acq_df = pd.concat([get_acq_session(info, pos_df) for info in info_df.reset_index().itertuples()])

