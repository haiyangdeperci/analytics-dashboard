# think about how to handle lists with assignees
# choice: total & per team member + avg
# get three data metrics first, then add tot / avg / per tm
# later - way to store the data and periodically refresh it
# Timeframe - 1 August 2019 - now (14 Jan 2021)
# number of bugs - line chart (over time - ordered) - open + currently X open bugs
# (potentially: start point, close point for each issue on a grid)
# number of issues - total & good first issues over time (area chart) + currently X% of GFI
# * time to close bugs - histogram (days)
# * pie chart - current number of open bugs per team member
# for pull requests: grid with events:
# [issue on y, time on x, events (start, labelled, comment closed) as scatter]

# ADD Metric class that would be fed with data and created in the same manner

import time; time_start = time.time() # noqa
import streamlit as st
import requests
import os
import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
st.title('Analytics Dashboard')

GITHUB_API_REPO_URL = 'https://api.github.com/repos/activeloopai/Hub/'
GITHUB_API_ORG_URL = 'https://api.github.com/orgs/activeloopai/'
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
__api_calls__ = 0

START_DATETIME = pd.Timestamp(year=2019, month=8, day=1, tz='UTC')
NOW = pd.Timestamp.now(tz='UTC')


def getData(base, path, params=None):
    # will need to implement pagination
    # this should be changed when a dashboard class is created
    global __api_calls__
    __api_calls__ += 1
    params = params or {}
    params['per_page'] = 100
    params['page'] = 1
    results = []
    breakpoint()
    while r := requests.get(
        f'{base}{path}', params=params,
        headers={'Authorization': f'token {GITHUB_TOKEN}'}
    ).json():
        params['page'] += 1
        results += r
        if params['page'] > 20:
            raise Exception
    return results


def parseNestedData(observation, key):
    if isinstance(observation, list):
        return [d[key] for d in observation]
    if observation:
        return observation[key]


def isPullRequest(issue):
    if issue:
        return True
    return False


def getWhitelist():
    members = getData(GITHUB_API_ORG_URL, 'members')
    members = pd.DataFrame(members)
    return members.login


def getCommits():
    commits = getData(GITHUB_API_REPO_URL, 'commits')
    commits = pd.DataFrame(commits)
    return commits


def getIssues(state='all', labels=None):
    params = {'state': state}
    if labels:
        params['labels'] = labels
    bugs = getData(GITHUB_API_REPO_URL, 'issues', params)
    bugs = pd.DataFrame(bugs)
    # restrict columns to cols if they exist
    cols = [
        'id', 'node_id', 'number', 'title', 'user', 'state',
        'locked', 'assignees', 'comments', 'created_at',
        'updated_at', 'closed_at', 'author_association',
        'pull_request'
    ]
    if not labels:
        cols.append('labels')
    intersected = bugs.columns.intersection(cols)
    if 'pull_request' in intersected:
        bugs['pull_request'] = pd.isnull(bugs['pull_request'])
        bugs = bugs[bugs['pull_request']]
    bugs = bugs[intersected]
    bugs['author_id'] = bugs.user.apply(parseNestedData, args=('id',))
    bugs['author_name'] = bugs.user.apply(parseNestedData, args=('login',))
    bugs['assignee_ids'] = bugs.assignees.apply(parseNestedData, args=('id',))
    bugs['assignee_names'] = bugs.assignees.apply(
        parseNestedData, args=('login',)
    )
    bugs.drop(['user', 'assignees'], 1, inplace=True)
    bugs[['created_at', 'updated_at', 'closed_at']] = bugs[[
        'created_at', 'updated_at', 'closed_at'
    ]].apply(pd.to_datetime, utc=True)
    return bugs


def overTime(data):
    dr = pd.DataFrame({'key': -1, 'date': pd.date_range(
        # workaround for the bug
        # https://github.com/pandas-dev/pandas/issues/31930
        START_DATETIME,
        # ^^ data.created_at.min(),
        NOW, freq='D'
    )})
    data['key'] = -1
    data = data[['number', 'created_at', 'closed_at', 'key']]
    merged = pd.merge(dr, data, on='key')
    merged['count'] = np.where(
        (merged.date > merged.created_at) & (
            (merged.date < merged.closed_at) | pd.isnull(merged.closed_at)
        ),
        True, False
    )

    nOverTime = merged[merged['count']].groupby('date')['count'].count()
    nOverTime = pd.concat(
        [dr.set_index('date'), nOverTime], 1).drop('key', 1).fillna(0)
    return nOverTime


def total_hours(timedelta):
    return round(timedelta.total_seconds() / 60 / 60, 2)


def periodically(data, f='mean'):
    # thresholds = [7, 30, 91]  # week, month, quarter
    # periods = ['weekly', 'monthly', 'quarterly']
    thresholds = {'weekly': 7, 'monthly': 30, 'quarterly': 91}
    metrics = []
    for name, t in thresholds.items():
        current = NOW - pd.Timedelta(days=t), NOW
        last = tuple(
            dt - pd.Timedelta(days=t) for dt in current
        )
        current_metric = data[slice(*current)].agg(f).squeeze()
        last_metric = data[slice(*last)].agg(f).squeeze()
        diff_metric = (current_metric - last_metric) / last_metric
        metrics.append({
            'period': name,
            f: current_metric,
            'diff (%)': diff_metric * 100
        })
    return metrics


def groupPer(data, per_what, name, na=None):
    _per = pd.DataFrame(data[per_what].tolist()).stack()
    _per.index = _per.index.droplevel(-1)
    _per.name = name
    joined = data.reset_index().join(_per)
    if na:
        joined = joined.fillna(na)
    return joined.groupby(name)


class Issues():
    def __init__(self, data):
        self.data = data
        self.data = self.data.reset_index()
        self.underlying_metric = Metric(self.data)
        self.getLabels()
        self.generateLabelledMetrics()
        self.metrics = {**self.label_data}
        self.generateCommunityMetrics()

    def getLabels(self):
        self.data['label_names'] = self.data.labels.apply(
            parseNestedData, args=('name',))

    def labels(self):
        return self.data['label_names']

    def generateLabelledMetrics(self):
        stacked = pd.DataFrame(self.data.label_names.tolist()).stack()
        stacked.index = stacked.index.droplevel(-1)
        label_names = stacked.unique()
        self.label_data = dict(zip(label_names, np.zeros(len(label_names))))

        for label in label_names:
            print(label)
            self.label_data[label] = Metric(self.data.iloc[
                stacked[stacked == label].index],
                underlying_metric=self.underlying_metric)

    def generateCommunityMetrics(self):
        self.metrics['community'] = Metric(self.data[
            ~self.data.author_name.isin(whitelist)],
            underlying_metric=self.underlying_metric)

# issue_community_percentages = issues[
#     'author_is_community'].value_counts() / len(issues) * 100

        # return label_data


class Metric():
    def __init__(self, data, underlying_metric=None, derived_from='issue'):
        self.data = data
        self.underlying_metric = underlying_metric
        self.derived_from = derived_from
        self.thresholds = {'Week': 7, 'Month': 30, 'Quarter': 91}
        if self.derived_from == 'issue':

            self.open_data, self.closed_data = self.splitState()

        # max n of days in 2 consecutive quarters
        # self.time_frame = self.time_frame.reset_index().tail(184)
        self.last_2Q_data = self.data[
            self.data.created_at > NOW - pd.Timedelta(days=184)]
        self.time_frame = self.overTime(self.last_2Q_data).reset_index()
        self.deriveTimePeriods(self.time_frame, 'date')
        self.deriveTimePeriods(
            self.last_2Q_data, 'created_at', prefix='created_at.')
        self.generateSeries()

    def generateTimeFrame(self):
        pass

    def generateSeries(self):
        # 'Count', 'Proportion', 'Last_30_day_average', 'This_Week_average',
        # 'WeekTD', 'Prev_WeekTD', 'Week_dynamics', 'This_Month_average',
        # 'MonthTD', 'Prev_MonthTD', 'Month_dynamics', This_Quarter_average',
        # 'QuarterTD', 'Prev_QuarterTD', 'Quarter_dynamics'
        self.series = pd.Series()
        self.setCount()
        self.setProportion()
        self.setLast30Days()
        self.setPeriodicData()

    def setPeriodicData(self):
        curWMQ = self.time_frame[
            list(self.thresholds)].iloc[-1]
        lastWMQ = pd.Series()
        for period in self.thresholds:
            lastWMQ[period] = self.time_frame.loc[self.time_frame[
                self.time_frame[period] == curWMQ[period]
            ].first_valid_index() - 1][period]

            self.series[f'This_{period}_average'] = self.time_frame[
                self.time_frame[period] == curWMQ[period]]['count'].mean()

            self.series[f'{period}TD'] = self.getPeriodToDate(
                self.last_2Q_data,
                f'created_at.{period}', curWMQ[period]
            )

            self.series[f'Prev_{period}TD'] = self.getPeriodToDate(
                self.last_2Q_data,
                f'created_at.{period}', lastWMQ[period]
            )
            self.setDynamics(period)

    def setLast30Days(self):
        self.series['Last_30_day_average'] = self.time_frame.tail(30)[
            'count'].mean()

    def setCount(self):
        self.series['Count'] = len(self)

    def setProportion(self):
        self.series['Proportion'] = self.ratio(True)

    def _dynamics(self, current, last):
        try:
            return (current - last) / last
        except ZeroDivisionError:
            if current > last:
                return float('inf')
            return np.nan

    def setDynamics(self, period):
        self.series[f'{period}_dynamics'] = self._dynamics(
            self.series[f'{period}TD'], self.series[f'Prev_{period}TD']
        )

    def __len__(self):
        if self.derived_from == 'issue':
            return len(self.open_data)
        return len(self.data)

    def ratio(self, stringify=False):
        if self.underlying_metric is not None:
            r = len(self) / len(self.underlying_metric)
            return f'{r:.1%}' if stringify else r
        return np.nan

    def splitState(self):
        mask = self.data.state == 'open'
        return self.data[mask], self.data[~mask]

    def deriveTimePeriods(self, frame, date_col, prefix=''):
        frame[prefix + 'Week'] = frame[date_col].dt.week
        frame[prefix + 'Month'] = frame[date_col].dt.month
        frame[prefix + 'Quarter'] = frame[date_col].dt.quarter

    def getPeriodToDate(self, data, date_col, period_metric):
        return np.int64(len(data[data[date_col] == period_metric]))

    # this needs to be in parsing
    def deriveResolutionMetric(self):
        self.data[
            'actual_resolution_time'
        ] = self.data.closed_at - self.data.created_at
        self.data[
            'predicted_resolution_time'
        ] = self.data.closed_at.fillna(NOW) - self.data.created_at

    def overTime(self, data):
        dr = pd.DataFrame({'key': -1, 'date': pd.date_range(
            # workaround for the bug
            # https://github.com/pandas-dev/pandas/issues/31930
            START_DATETIME,
            # ^^ data.created_at.min(),
            NOW, freq='D'
        )})
        data['key'] = -1
        merged = pd.merge(dr, data, on='key')
        data.drop('key', 1, inplace=True)
        if 'closed_at' not in merged:
            merged['closed_at'] = pd.Series(
                pd.NaT, dtype='datetime64[ns, UTC]')
        merged['count'] = np.where(
            (merged.date > merged.created_at) & (
                (merged.date < merged.closed_at) | pd.isnull(merged.closed_at)
            ),
            True, False
        )

        nOverTime = merged[merged['count']].groupby('date')['count'].count()
        nOverTime = pd.concat(
            [dr.set_index('date'), nOverTime], 1).drop('key', 1).fillna(0)
        return nOverTime

    def periodically(self, f='mean'):
        # thresholds = [7, 30, 91]  # week, month, quarter
        # periods = ['weekly', 'monthly', 'quarterly']
        metrics = []
        for name, t in self.thresholds.items():
            current = NOW - pd.Timedelta(days=t), NOW
            last = tuple(
                dt - pd.Timedelta(days=t) for dt in current
            )
            current_metric = self.data[slice(*current)].agg(f).squeeze()
            last_metric = self.data[slice(*last)].agg(f).squeeze()
            diff_metric = (current_metric - last_metric) / last_metric
            metrics.append({
                'period': name,
                f: current_metric,
                'diff (%)': diff_metric * 100
            })
        return metrics


class CommitData(Metric):
    def __init__(self, data):
        self.data = data
        commit_author = self.data['commit'].apply(
            parseNestedData, args=('author',))
        self.data['commit_author_name'] = commit_author.apply(
            parseNestedData, args=('name',))
        self.data['author_login'] = self.data['author'].apply(
            parseNestedData, args=('login',))
        self.data['created_by'] = self.data['author_login'].fillna(
            self.data['commit_author_name'])
        # eventually get the names from the whitelist and map
        self.extra_committers = {
            'Vagharsh Kandilian': 'vagharsh',
            'Edward Grigoryan': 'edogrigqv2',
            'Jason Ge': 'jasonge27',
            # 'Gegham Vardanyan': '',
            # 'Gegham': ''
        }
        self.data['created_by'] = self.data.created_by.replace(
            self.extra_committers)
        self.data[
            'author_is_community'] = ~self.data.created_by.isin(whitelist)
        self.data['created_at'] = commit_author.apply(
            parseNestedData, args=('date',)).apply(pd.to_datetime, utc=True)
        self.underlying_metric = self.data.copy()  # all the commits
        self.data = self.data[self.data.author_is_community]  # comts w/o team
        super().__init__(self.data, self.underlying_metric, 'commit')

open_bugs = getIssues('open', 'bug')
st.write("Number of open bugs:", len(open_bugs))

open_issues = getIssues('open')
open_good_first_issues = getIssues('open', 'good first issue')
st.write(
    "Number of good first issues / all issues:",
    f'{len(open_good_first_issues)} / {len(open_issues)} '
    f'({len(open_good_first_issues) / len(open_issues):.1%})',
)

closed_bugs = getIssues('closed', 'bug')
closed_bugs[
    'issue_resolution_time'
] = closed_bugs['closed_at'] - closed_bugs['created_at']

st.write(
    'Average amount of time to close a bug ticket:',
    closed_bugs.issue_resolution_time.mean().round('s')
)


closed_bugs['issue_resolution_time'] = closed_bugs[
    'issue_resolution_time'
].astype(np.int64)
issue_resolution_time_per_assignee = pd.to_timedelta(
    groupPer(
        closed_bugs, 'assignee_names',
        '_assignee', 'No Assignee'
    )['issue_resolution_time'].mean()
)

st.write(
    'Average amount of time to close a bug ticket per assignee (hours):'
)
st.bar_chart(
    issue_resolution_time_per_assignee.apply(total_hours).sort_values()
)

bugs = pd.concat([open_bugs, closed_bugs])
n_bugs = overTime(bugs)
st.write('Number of open bugs over time:')
st.line_chart(n_bugs)

n_bugs_periodic_info = periodically(n_bugs)
st.write(pd.DataFrame(periodically(n_bugs)))
st.write(bugs)
# st.write(
#     'Open bugs - {0.name}: {0.metric:.2f}; monthly: {:.2f}; quarterly: {:.2f}'.format(
#         *periodically(n_bugs))
# )


# this currently shows graphs for currently opened issues
# should be restructured to include the closed issues as well
issues = getIssues()
good_first_issues = getIssues(labels='good first issue')
n_issues = overTime(issues)
n_good_first_issues = overTime(good_first_issues)
n_i_gfi = pd.concat([
    n_issues.add_prefix('issue.'),
    n_good_first_issues.add_prefix('good_first_issue.')], 1)
# this graph tells you what how many good first issues
# (defined as of now) there are out of all
# the assumption is that an issue marked GFI was always a GFI
# an alternative: tracking when the issue was labelled and became GFI
# this graph stops at midnight today UTC
st.write(
    'Number of open good first issues (as currently labelled) '
    'juxtaposed with the total number of opened issues at a point in time:'
)
st.line_chart(n_i_gfi)

open_bugs_per_assignee = groupPer(
    open_bugs, 'assignee_names', '_assignee', 'No Assignee'
)['number'].count()

st.write('Open bugs per assignee')
fig = px.pie(
    names=open_bugs_per_assignee.index,
    values=open_bugs_per_assignee.values
)  # , values='Tweets', names='Sentiment'
st.plotly_chart(fig)

# issues.author_association.value_counts() / len(issues)

### REFACTOR
whitelist = getWhitelist()

# issues['author_is_community'] = ~issues.author_name.isin(whitelist)
# issue_community_percentages = issues[
#     'author_is_community'].value_counts() / len(issues) * 100
# st.write(
#     'Percetange of issues opened by the community',
#     issue_community_percentages)


# commits = commits.dropna()
# commits['author_name'] = commits.author.apply(parseNestedData, args=('login',))
# commits['author_is_community'] = ~commits.author_name.isin(whitelist)
# commit_community_percentages = commits[
# #     'author_is_community'].value_counts() / len(commits) * 100
# st.write(
#     'Percetange of commits from the community',
#     commit_community_percentages)

###

period_choice = st.sidebar.date_input(
    'Pick a date range to filter through',
    [START_DATETIME, NOW],
    min_value=START_DATETIME,
    max_value=NOW
)
assignee_choice = st.sidebar.selectbox(
    'Pick a team member to filter through',
    pd.Series([None]).append(whitelist, ignore_index=True)
)

commits = getCommits()
commit_data = CommitData(commits).series
issue_metrics = Issues(getIssues()).metrics
metrics = pd.DataFrame([issue_metrics[m].series for m in issue_metrics] + [commit_data])

st.write(metrics)
# all_gfi = label_data['good first issue']
# all_bugs = label_data['bug']

st.write('Dashboard load time:', time.time() - time_start)
