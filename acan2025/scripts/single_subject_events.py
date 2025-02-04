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
# "rawdata/sub-danger-ses-RR20.03" folder to be in the parent directory
# of the current location, with the events file inside it.
#
# We can break up long strings across lines be enclosing them in
# parentheses.

# %%
events_fn = ('../rawdata/sub-danger/ses-RR20.03/'
             'sub-danger_ses-RR20.03_task-RR20_acq-A_events.csv')

# %%
events = pd.read_csv(events_fn)
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

# %% [markdown]
#
# The event_rates table now has just a single column of data, the rate
# of the event in 30 second windows, with the type of event on the left
# as an index column. If we wanted to select out just the rate of mag
# entries, for example, we could use the `.loc` method.
#
# The loc method allows us to select data by label, with one entry for
# the row labels and one for the column labels of interest. In this
# case, because each row has two labels ('event_id' and 'onset'), we can
# use a `tuple` to select the rows. `slice(None)` matches any value in
# the column.

#  %%
event_rates.loc[('mag', slice(None)), :]

# %% [markdown]
#
# We can use the hvplot library to create an interactive plot of the
# data. the `by` argument separates out the data based on the specified
# column. In this case, we are separating the data by the event_id,
# giving us a new line for each event type.

# %%
event_rates.hvplot.line(x='onset', y='rate', by='event_id')

# %% [markdown]
#
# What if we wanted both acquisition sessions? We could load them into
# separate variables, or into a list of DataFrames. But we could also
# just load them into a single dataframe and identify each acquision
# session with an extra column.
#
# Let's write a small function that returns our events DataFrame given a
# filename.

# %%
def load_events(filename):
    events = pd.read_csv(filename)
    events['onset'] = pd.to_timedelta(events['onset'], unit='s')
    events = events.fillna(0.01).set_index('onset')
    return events

# %% [markdown]
#
# We can call the load_events function using our filename from earlier
# to test it.

# %%
load_events(events_fn)

# %% [markdown]
# Now we can use that function to load both acquisition sessions.
#
# A Path object can be used to represent a folder on the filesystem.

# %%
ses_path = Path('../rawdata/sub-danger/ses-RR20.03')
events_filenames = {'A': load_events(ses_path / 'sub-danger_ses-RR20.03_task-RR20_acq-A_events.csv'),
                    'B': load_events(ses_path / 'sub-danger_ses-RR20.03_task-RR20_acq-B_events.csv')}
events_filenames

# %% [markdown]
#
# {id: value} creates a "dictionary" object, which allows us to access
# values by keywords.
#
# Try running `events_filenames['A']`

# %%


# %% [markdown]
#
# We can use the `pd.concat` function to combine the two DataFrames into
# a single DataFrame, with an extra column to identify the acquisition
# session.

# %%
ses_events = pd.concat(events_filenames, names=['acq'])
ses_events

# %% [markdown]
#
# Now we can run the same rolling window analysis as before, but instead
# of just grouping by the event_id, we also group by the acquisition
# session.

# %%
ses_event_rates = (
    ses_events
        .reset_index('acq')
        .groupby(['acq', 'event_id'])
        .rolling(pd.to_timedelta('30s'))
        .count()
        .rename(columns={'duration': 'rate'}))
ses_event_rates

# %% [markdown]
#
# Now let's use the seaborn library to create a faceted plot of the data.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='darkgrid')
sns.relplot(data=ses_event_rates.reset_index(),
            x='onset',
            y='rate',
            row='event_id',
            hue='acq',
            kind='line',
            alpha=0.75,
            aspect=3)

# %% [markdown]
#
# Now can we extend this to both pre-reversal and post-reversal
# sessions?
#
# If we don't want to manually enter all the filenames, we can use the
# `glob` method of the Path object to search for files that match a
# pattern.
#
# e.g. let's find all events files in the session folder.

# %%
ses_events_paths = [str(p) for p in ses_path.glob('*_events.csv')]
ses_events_paths
