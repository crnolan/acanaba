# %%
import pandas as pd
from pathlib import Path
import hvplot.pandas

# %%
data_path = Path('../rawdata/sub-angelina/ses-RR20.03')

# %%
events_paths = [str(p) for p in data_path.glob('*_events.csv')]
events_paths

# %%
events = pd.read_csv(events_paths[0])
events['onset'] = pd.to_timedelta(events['onset'], unit='s')
events = events.fillna(0.01).set_index('onset')
events

# %% [markdown]
#
# The formatting below is just to make the "method chaining" clear and
# avoid excessively long lines. One could just write the first
# assignment as:
#
# ```python
# event_rates = events.groupby('event_id').rolling(pd.to_timedelta('30s')).count().rename(columns={'duration': 'rate'}))
# ```
#
# but it's much harder to read and quite possibly requires horizontal
# scrolling to read the whole line.

# %%
event_rates = (
    events
        .groupby('event_id')
        .rolling(pd.to_timedelta('30s'))
        .count()
        .rename(columns={'duration': 'rate'}))
event_rates

# %%
event_rates.hvplot.line(x='onset', y='rate', by='event_id')

# %% [markdown]
#
# What if we wanted both acquisition sessions? We could load them into
# separate variables, or into a list of DataFrames. But we could also
# just load them into a single dataframe and identify each acquision
# session with an extra column.

# %%

