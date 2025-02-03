# %% [markdown]
#
# # Exploring event data
#
# This script is a simple overview of how to load event data from
# sessions in a single animal.

# %% [markdown]
#
# We need to load some libraries to help us work with our data.
#
# pandas: A library for working with data in a table format. If you are
#         familiar with R, pandas is similar to the data.frame object.
# pathlib: A library for working with file paths.
# hvplot: A library for creating interactive plots. For static /
#         print-ready plots, or for more complex faceted data plots, the
#         seaborn and/or matplotlib libraries might be more appropriate.

# %%
import pandas as pd
from pathlib import Path
import hvplot.pandas

# %% [markdown]
#
# We can refer to a path on the filesystem either relative to the
# current location or absolutely. The path object below is expecting the
# "rawdata/sub-angelina-ses-RR20.03" folder to be in the parent
# directory of the current location.

# %%
data_path = Path('../rawdata/sub-angelina/ses-RR20.03')

# %% [markdown]

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

