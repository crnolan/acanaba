import json
from pathlib import Path
from collections import namedtuple
import re
import numpy as np
import pandas as pd
from typing import Union
import logging


def load_recordings(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load recordings from a data path.

    Args:
        data_path: The path to the data folder.

    Returns:
        A `pd.DataFrame` containing the recordings.
    """
    data_path = Path(data_path)
    glob_pattern = 'sub-*/ses-*/sub-*_ses-*_task-*_acq-*.json'
    re_pattern = (r'sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+).json')
    json_files = list(data_path.glob(str(glob_pattern)))
    extracted_data = []
    for json_path in json_files:
        match = re.search(re_pattern, str(json_path.name))
        if match:
            # Pull out the json data
            try:
                with open(json_path) as f:
                    recording = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f'Error reading {json_path.name}')
                continue

            # Verify json has the required info
            required_keys = set(['sub', 'ses', 'acq', 'block',
                                 'firston', 'laston', 'firstmed', 'lastmed'])
            if not required_keys.issubset(recording.keys()):
                logging.warning(
                    f'Missing data in {json_path.name}, need keys {required_keys}')
            sub, ses, task, acq = match.groups()
            if (sub != recording['sub'] or ses != recording['ses']
                or acq != recording['acq']):
                logging.warning(f'Session metadata is inconsistent between '
                                f'path and JSON at {json_path.name}')
            recording['task'] = task
            prefix = f'sub-{sub}_ses-{ses}_task-{task}_acq-{acq}'

            # Find corresponding DLC and events files
            dlc_glob = list(json_path.parent.glob(prefix + '*_filtered.h5'))
            if len(dlc_glob) > 1:
                raise ValueError(
                    f"Multiple DLC files found for {json_path.name}")
            elif len(dlc_glob) == 0:
                logging.warning(f'No DLC file found for {json_path.name}')
                recording['dlc_path'] = None
            else:
                recording['dlc_path'] = dlc_glob[0]
            events_path = (json_path.parent / (prefix + '_events.csv'))
            if not events_path.exists():
                logging.warning(f'No events file found for {json_path.name}')
                events_path = None
            recording['events_path'] = events_path
            extracted_data.append(recording)
    recordings = pd.DataFrame(extracted_data)
    recordings = recordings.rename(columns={'sub': 'subject', 'ses': 'session'})
    recordings = recordings.set_index(['subject', 'session', 'task', 'acq'])
    return recordings


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


def load_events_session(events_path: Union[str, Path],
                        block: str) -> pd.DataFrame:
    '''Load one session analysed by DeepLabCut into a DataFrame'''
    try:
        df = pd.read_csv(events_path)
    except FileNotFoundError:
        print(f'File not found: {events_path}')
        return None
    df['onset'] = pd.to_timedelta(df['onset'], unit='s')
    if block:
        df = df.replace({'lp': 'llp' if block[0] == 'L' else 'rlp'})
    return df.set_index('onset')
