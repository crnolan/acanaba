# %%
import pandas as pd
from pathlib import Path
import re
import hvplot.pandas
import logging
import seaborn as sns
from statannotations.Annotator import Annotator
from utils import load_recordings, load_track_session, load_events_session

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
idx = pd.IndexSlice

# %%
data_path = Path('../rawdata')
rec_cols = ['subject', 'session', 'task', 'acq']

# %%
recordings = load_recordings(data_path)
recordings

# %%
#
# Get events data.
events_df = (recordings[['events_path', 'block']]
             .groupby(rec_cols).apply(
                lambda x: load_events_session(*(x.iloc[0])))
             .fillna(0.1))


# %%
#
# Let's get some preference data based on RR20 sessions
preferences = (
    events_df
    .loc[idx[:, :, ['RR20'], :, :], :]
    .query('event_id.isin(["llp", "rlp"])')
    .reset_index(['subject', 'session', 'task', 'acq'])
    .groupby(['subject', 'session', 'task', 'acq'])['duration']
    .apply(lambda x: x.count() / (x.index.max() / pd.to_timedelta('1s')))
    .rename('rate')
    .to_frame()
    .join(recordings['block'].str[1])
    .set_index('block', append=True)
    .droplevel('acq'))
preferences

# %%
# Ratio of preferences for purified vs. grain
preference_ratio = (
    preferences
    .unstack('block')
    .droplevel(0, axis=1)
    .apply(lambda x: x['P'] / x['G'], axis=1))
preference_ratio

# %%
#
# Now get a summary of the extinction as a ratio to pre-deval
extinction_weighted = (
    events_df
    .loc[idx[:, :, ['deval01'], :, :], :]
    .query('event_id.isin(["llp", "rlp"])')
    .reset_index(['subject', 'session', 'task', 'acq'])
    .groupby(['subject', 'session', 'task', 'acq', 'event_id'])['duration']
    .apply(lambda x: x.count() / (x.index.max() / pd.to_timedelta('1s')))
    .rename('rate')
    .to_frame()
    .join(recordings['block'])
    .assign(outcome=lambda x: x.block.str[1]
                    if x.index.get_level_values('event_id').str[0] == x.block[0].lower()
                    else 'N'))
    # .assign(outcome=lambda x: x.block.str[1]
    #                           if x.index.get_level_values('event_id').str[0] == x.block[0].lower()
    #                           else 'G' if x.block[0] == 'P' else 'P'))
# extinction_weighted['day'] = extinction_weighted.index.get_level_values('session').str[-2:].astype(int)
extinction_weighted

# %%
#
# First let's look at extinction over deval sessions
extinction_rates = (
    events_df
    .loc[idx[:, :, ['deval01', 'deval02'], :, :], :]
    .query('event_id.isin(["llp", "rlp"])')
    .reset_index(['subject', 'session', 'task', 'acq'])
    .groupby(['subject', 'session', 'task', 'acq'])['duration']
    .apply(
        lambda x: (
            x
            .notna()
            .resample(pd.to_timedelta('5s')).sum()
            .rolling(pd.to_timedelta('30s')).sum()
            .sort_index()),
        include_groups=False)
    .rename('rate')
    .to_frame())
extinction_rates['day'] = extinction_rates.index.get_level_values('session').str[-2:].astype(int)
extinction_rates

# %%
sns.set_theme(style='darkgrid')
sns.relplot(data=extinction_rates.reset_index(),
            x='onset',
            y='rate',
            row='task',
            col='day',
            hue='subject',
            kind='line',
            alpha=0.75,
            aspect=3)

# %%
#
# And summary
extinction_summary = (
    events_df
    .loc[idx[:, :, ['deval01', 'deval02'], :, :], :]
    .query('event_id.isin(["llp", "rlp"])')
    .reset_index(['subject', 'session', 'task', 'acq'])
    .groupby(['subject', 'session', 'task', 'acq'])['duration']
    .count()
    .rename('rate')
    .to_frame())
extinction_summary['day'] = extinction_summary.index.get_level_values('session').str[-2:].astype(int)
extinction_summary

# %%
sns.violinplot(data=extinction_summary.reset_index(),
            x='task',
            y='rate',
            hue='day',
            cut=0,
            alpha=0.5,
            inner='point')

# %%
ax = sns.stripplot(x='task',
                   y='rate',
                   hue='day',
                   linestyles='',
                   dodge=.3,
                   data=extinction_summary.reset_index())
for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
    ax.plot([x0, x1], [y0, y1], color='black', ls=':', zorder=0)


# %%
#
# Let's use day 2 / day 1 rates as an index of extinction
extinction_index = (
    extinction_summary
    .set_index('day', append=True)
    .reset_index('session', drop=True)
    .unstack('day')
    .apply(lambda x: x.loc[('rate', 2)] / x.loc[('rate', 1)], axis=1)
)
extinction_index.unstack('task')

# %%
event_rates = (
    events_df
    .reset_index(['subject', 'session', 'task', 'acq'])
    .groupby(['subject', 'session', 'task', 'acq', 'event_id'])
    .apply(
        lambda x: (
            x
            .notna()
            .resample(pd.to_timedelta('5s')).sum()
            .rolling(pd.to_timedelta('30s')).sum()
            .sort_index()),
        include_groups=False)
    .rename(columns={'duration': 'rate'}))
event_rates

# %%
#
(events_df
    .loc[idx[:, :, 'deval02', :]]
    .groupby(['subject', 'session', 'event_id'])
    .count()
    .join(recordings['block']))

# %%

# Get deval counterbalancing data (add to JSONs)

# %%
# projpath = Path('..')

# glob_pattern = 'rawdata/sub-*/ses-deval*/sub-*_ses-*_task-*_acq-*_events.csv'
# re_pattern = (r'sub-([^_]+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_events.csv')
# fns = list(projpath.glob(str(glob_pattern)))
# extracted_data = {}
# for fn in fns:
#     match = re.search(re_pattern, str(fn.name))
#     if match:
#         try:
#             events = pd.read_csv(fn)
#         except pd.errors.EmptyDataError:
#             logging.warning(f'Error reading {fn.name}')
#             continue
#         events.set_index('onset', inplace=True)
#         extracted_data[match.groups()] = events

# events = pd.concat(extracted_data)
# events

# %%
