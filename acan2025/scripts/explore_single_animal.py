# %%
import pandas as pd
from pathlib import Path

# %%
data_path = Path('../rawdata/sub-angelina/ses-RR20.03')

# %%
events_path = [str(p) for p in data_path.glob('*A_events.csv')][0]
events_path

# %%
events = pd.read_csv(events_path)
events['onset'] = pd.to_timedelta(events['onset'], unit='s')
events = events.set_index('onset')
events

# %%
events.groupby('event_id').rolling(pd.to_timedelta('60s')).count()

# %%
events.loc[events['event_id'] == 'lp'].count()

# %%
