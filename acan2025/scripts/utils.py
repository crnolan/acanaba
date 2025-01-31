import json
from pathlib import Path
from collections import namedtuple
import re
import numpy as np
import pandas as pd
from typing import Union


Recording = namedtuple(
    "Recording", ["subject", "session", "task", "acq", "med_path", "dlc_path"])


def load_recordings(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load recordings from a data path.

    Args:
        data_path: The path to the data folder.

    Returns:
        A `pd.DataFrame` containing the recordings.
    """
    data_path = Path(data_path)
    glob_pattern = 'sub-*/ses-*/sub-*_ses-*_task-*_acq-*_events.csv'
    re_pattern = (r'sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_'
                  r'events.csv')
    data_files = list(data_path.glob(str(glob_pattern)))
    extracted_data = []
    for file_path in data_files:
        match = re.search(re_pattern, str(file_path.name))
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
