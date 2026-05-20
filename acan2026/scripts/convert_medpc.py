# %%
import behapy.medpc as medpc
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import re

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# %%
projroot = Path("..")
choice_config = json.load(open(projroot / 'etc/choice_medpc_config.json'))
train_config = json.load(open(projroot / 'etc/train_medpc_config.json'))
for config in (choice_config, train_config):
    config['event_map'] = {int(key): value
                            for key, value in config['event_map'].items()}
regex_pattern = (r"sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_rawmed.txt")
msn_pattern = (r'[^ ]+ (L|R) (PUR|GRN) (RR20|RR10|RR5|CRF)')
overwrite = True

# %%
for fn in (projroot / 'rawdata').glob('**/*rawmed.txt'):
    match = re.search(regex_pattern, str(fn.name))
    if not match:
        logging.warning(f'Bad template match for {fn.name}')
        continue
    sub, ses, task, acq = match.groups()
    events_fn = fn.parent / f'sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_events.csv'
    logging.info(f'Processing {fn.name}')
    if 'deval' in ses:
        config = choice_config
    else:
        config = train_config
    variables = medpc.parse_file(str(fn))
    info = medpc.experiment_info(variables)
    events = medpc.get_events(variables[config['timestamp']],
                              variables[config['event_index']],
                              config['event_map'])
    if events.empty:
        logging.warning(f'No events found for {fn.name}')
        events_df = pd.DataFrame({
            'onset': [],
            'duration': [],
            'event_id': []
        })
    else:
        events_df = pd.DataFrame({
            'onset': events.index / pd.Timedelta('1s'),
            'duration': np.nan,
            'event_id': events['event_id']
        })
    if events_fn.exists() and not overwrite:
        logging.info(f'Events file already exists for {fn.name}')
    else:
        events_df.to_csv(events_fn, index=False)

    match_msn = re.search(msn_pattern, str(info['MSN']))
    block = None
    if not match_msn:
        if acq != 'deval':
            logging.warning(f'Bad MSN match for {fn.name}')
    else:
        block = match_msn.groups()[0] + match_msn.groups()[1][0]
    info_fn = fn.parent / f'sub-{sub}_ses-{ses}_task-{task}_acq-{acq}.json'
    info = {
        'sub': sub,
        'ses': ses,
        'acq': acq,
        'block': block,
        'leverin': None,
        'leverout': None,
    }
    if info_fn.exists() and not overwrite:
        logging.info(f'Info file already exists for {fn.name}')
    else:
        with open(info_fn, 'w') as f:
            json.dump(info, f, indent=4)
        print(info)

# %%
