# %%
import behapy.medpc as medpc
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import re

# %%
projroot = Path('..')
choice_config = json.load(open(projroot / 'etc/choice_medpc_config.json'))
train_config = json.load(open(projroot / 'etc/train_medpc_config.json'))
for config in (choice_config, train_config):
    config['event_map'] = {int(key): value
                            for key, value in config['event_map'].items()}
regex_pattern = (r"sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_rawmed.txt")

# %%
for fn in (projroot / 'rawdata').glob('**/*rawmed.txt'):
    match = re.search(regex_pattern, str(fn.name))
    if not match:
        logging.warning(f'Bad template match for {fn.name}')
        continue
    sub, ses, task, acq = match.groups()
    if 'deval' in ses:
        config = choice_config
    else:
        config = train_config
    variables = medpc.parse_file(fn)
    info = medpc.experiment_info(variables)
    events = medpc.get_events(variables[config['timestamp']],
                              variables[config['event_index']],
                              config['event_map'])
    events_df = pd.DataFrame({
        'onset': events['timestamp'] / pd.Timedelta('1s'),
        'duration': np.nan,
        'event_id': events['event']
    })
    events_df.to_csv(fn.parent / f'sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_events.csv', index=False)

# %%
