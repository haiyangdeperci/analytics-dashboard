import time; time_start = time.time() # noqa
import streamlit as st
import requests
import os
import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import pickle

load_dotenv()


START_DATETIME = pd.Timestamp(year=2019, month=8, day=1, tz='UTC')
NOW = pd.Timestamp.now(tz='UTC')


class DashboardException(Exception):
    pass


class TooManyAPICalls(DashboardException):
    pass


class BadArguments(DashboardException):
    pass


class RawData():
    _api_calls = 0

    def __init__(self, url):
        self.url = url

    def retrievePage(self, params):
        RawData._api_calls += 1
        if RawData._api_calls > 50:
            raise TooManyAPICalls

        return requests.get(
            self.url, params=params,
            headers=self.headers
        ).json()

    def getData(self, params=None, single=False):
        params = params or self.params
        params['per_page'] = 100
        params['page'] = 1
        results = []
        if single:
            return self.retrievePage(params)
        while r := self.retrievePage(params):
            params['page'] += 1
            results += r
        return results


class GithubData(RawData):
    def __init__(self, resource, subresource=None, *, params=None):
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
        self.headers = {
            'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
        self.base_url = 'https://api.github.com'
        self.org = 'activeloopai'
        self.repo = 'Hub'
        self.resource = resource
        self.subresource = subresource
        self.divisions = {
            'issues': 'repos',
            'commits': 'repos',
            'traffic': 'repos',
            'members': 'orgs'
        }
        self.params = params or {}
        self.url = self.build_url()

    def build_url(self):
        division = self.divisions[self.resource]
        url = f'{self.base_url}/{division}/{self.org}/'
        if division == 'repos':
            url += f'{self.repo}/'
        url = f'{url}{self.resource}'
        return f'{url}/{self.subresource}' if self.subresource else url


class BaseData():
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.data = pd.DataFrame(self.raw_data)

    @staticmethod
    def parseNested(observation, *keys):
        if len(keys) > 1:
            vals = []
            for key in keys:
                if isinstance(observation, list):
                    vals.append([d[key] for d in observation])
                else:
                    vals.append(observation[key])
            return pd.Series(vals)
        else:
            key = keys[0]
            if isinstance(observation, list):
                return [d[key] for d in observation]
            if observation:
                return observation[key]

    def __getattr__(self, name):
        return getattr(self.data, name)


class Data(BaseData):

    cache = {}

    def __init__(self, data, parse=True):
        self.parse_data = parse
        super().__init__(data)
        if self.parse_data:
            self.parse()

    @classmethod
    def whitelist(cls):
        if 'whitelist' in cls.cache:
            return cls.cache['whitelist']
        cls._members = BaseData(GithubData('members').getData()).login
        cls._extra_team = pd.Series(['mynameisvinn'])
        cls._members = cls._members.append(cls._extra_team)
        # add bots:
        cls._bots = pd.Series([
            'sourcery-ai[bot]',
            'dependabot-preview[bot]',
            'dependabot[bot]'
        ])
        cls._whitelist = cls._members.append(cls._bots, ignore_index=True)
        cls.cache['whitelist'] = cls._whitelist
        return cls._whitelist

    def __str__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)

    def detach(self, by=None, **kwargs):
        if by == 'community' and 'is_community' in self.data:
            return self.data[self.data.is_community]

    # def parse(self):
    #     self.data = self.data.sort_values('created_at').reset_index(drop=True)


class IssueData(Data):

    @staticmethod
    def isPullRequest(issue):
        if issue:
            return True
        return False

    def parse(self, removePullRequests=True, removeIssues=False):
        # restrict columns to cols if they exist
        cols = [
            'id', 'node_id', 'number', 'title', 'user', 'state',
            'locked', 'assignees', 'comments', 'created_at',
            'updated_at', 'closed_at', 'author_association',
            'pull_request', 'labels'
        ]

        cols = self.data.columns.intersection(cols)

        if removePullRequests and removeIssues:
            raise BadArguments
        if removeIssues:
            self.data = self.data[~pd.isnull(self.data['pull_request'])]
        if removePullRequests and 'pull_request' in cols:
            self.data['pull_request'] = pd.isnull(self.data['pull_request'])
            self.data = self.data[self.data['pull_request']]
        self.data = self.data[cols]
        self.data[[
            'author_name', 'author_id'
        ]] = self.data.user.apply(self.parseNested, args=('login', 'id'))
        self.data[[
            'assignee_names', 'assignee_ids'
        ]] = self.data.assignees.apply(self.parseNested, args=('login', 'id'))
        self.data.drop(['user', 'assignees'], 1, inplace=True)
        self.data[['created_at', 'updated_at', 'closed_at']] = self.data[[
            'created_at', 'updated_at', 'closed_at'
        ]].apply(pd.to_datetime, utc=True)
        self.data[
            'actual_resolution_time'
        ] = self.data.closed_at - self.data.created_at
        self.data[
            'predicted_resolution_time'
        ] = self.data.closed_at.fillna(NOW) - self.data.created_at
        self.data['is_community'] = ~self.data.author_name.isin(
            Data.whitelist()
        )
        self.data[
            'label_names'
        ] = self.data.labels.apply(self.parseNested, args=('name',))
        # super().parse()

    def detach(self, **kwargs):
        # ideally detach does not change data
        # but returns a new object
        by = kwargs.pop('by')
        if 'state' in kwargs:
            state = kwargs.pop('state')
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')
        else:
            labels = []

        if 'state' in by:
            self.data = self.data[self.data.state == state]

        if 'community' in by:
            self.data = super().detach(by=by, **kwargs)

        if 'label' in by:
            stacked = pd.DataFrame(self.data.label_names.tolist()).stack()
            stacked.index = stacked.index.droplevel(-1)
            all_label_names = pd.Index(stacked.unique())
            labels = all_label_names.intersection(labels)
            if labels.empty:
                labels = all_label_names
            label_data = {}
            for label in labels:
                label_data[label] = self.data.iloc[
                    stacked[stacked == label].index
                ]
            return label_data

        return self.data

    # not sure this is the right place
    def total_hours(self, timedelta):
        return round(timedelta.total_seconds() / 60 / 60, 2)


class CommitData(Data):

    def __init__(self, data):
        self.unlogged = {
            'Vagharsh Kandilian': 'vagharsh',
            'Edward Grigoryan': 'edogrigqv2',
            'Jason Ge': 'jasonge27',
            # 'Gegham Vardanyan': '',
            # 'Gegham': ''
        }
        super().__init__(data)

    def parse(self):
        cmtAutr = self.data['commit'].apply(self.parseNested, args=('author',))
        # fill with name where login is not available
        # and replace with login if known
        self.data[[
            'cm_name', 'created_at'
        ]] = cmtAutr.apply(self.parseNested, args=('name', 'date'))
        self.data['cm_login'] = self.data['author'].apply(
            self.parseNested, args=('login',)
        )
        self.data['created_by'] = self.data.cm_login.fillna(self.data.cm_name)
        print(type(self), dir(self))
        self.data['created_by'] = self.data.created_by.replace(self.unlogged)
        self.data['is_community'] = ~self.data.created_by.isin(
            Data.whitelist()
        )
        self.data['created_at'] = pd.to_datetime(
            self.data.created_at, utc=True
        )
        self.data['closed_at'] = pd.Series(pd.NaT, dtype='datetime64[ns, UTC]')
        # super().parse()

    def detach(self, **kwargs):
        # by community
        return super().detach(by='community', **kwargs)


class CommentData(Data):
    '''probably better to inhertic from some BaseIssueData
    because a lot of the code can be shared'''

    def __init__(self, data, issue_data):
        self.issue_data = issue_data
        issue_cols = ['created_at', 'author_name', 'is_community']
        self.issue_data = self.issue_data.set_index('number')[issue_cols]
        super().__init__(data)

    def parse(self):
        self.data[[
            'author_name', 'author_id'
        ]] = self.data.user.apply(self.parseNested, args=('login', 'id'))
        self.data.drop(['user'], 1, inplace=True)
        self.data[['created_at', 'updated_at']] = self.data[[
            'created_at', 'updated_at'
        ]].apply(pd.to_datetime, utc=True)
        # rename comment to closed_at as by commenting we are *closing*
        self.data = self.data.rename(columns={'created_at': 'closed_at'})
        self.data['number'] = pd.to_numeric(self.data.issue_url.str.rsplit(
            '/', 1, expand=True)[1])
        self.data['is_community'] = ~self.data.author_name.isin(
            Data.whitelist()
        )

        self.data = self.data.set_index('number').join(
            self.issue_data, rsuffix='_issue', how='right')
        self.data = self.data.reset_index().sort_values('closed_at')
        self.data['is_community'] = self.data.is_community.fillna(False)
        # this might go into parsing here b/c its required for res_time -verify
        # essentially it's leaving only first comments
        self.data = self.data[
            self.data.is_community_issue & ~self.data.is_community
        ].drop_duplicates('number')

        self.data[
            'actual_resolution_time'
        ] = self.data.closed_at - self.data.created_at
        self.data[
            'predicted_resolution_time'
        ] = self.data.closed_at.fillna(NOW) - self.data.created_at
        # add the issue create_at

    def detach(self, **kwargs):
        # get pull requests and issues separately - later

        by = kwargs.pop('by')
        if 'community' in by:
            self.data = super().detach(by=by, **kwargs)
        if 'first_comment' in by:
            pass

class VisitorData(Data):

    def __init__(self, data, save=False):
        self.storage_file = 'visitors.pkl'
        self.old_data = self.load()
        super().__init__(data)
        if self.old_data:
            # this could be improved to get data only X days
            self.data = self.old_data.append(
                self.data, ignore_index=True).drop_duplicates(keep='last')
        if save:
            self.save()

    def parse(self):
        self.data = pd.json_normalize(self.data.views)
        self.data['timestamp'] = pd.to_datetime(self.data.timestamp, utc=True)
        self.data = self.data.rename(columns={'timestamp': 'created_at'})
        self.data['closed_at'] = pd.Series(pd.NaT, dtype='datetime64[ns, UTC]')

    def load(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'rb') as handle:
                data = pickle.load(handle)
            return data

    def save(self):
        with open(self.storage_file, 'wb') as handle:
            pickle.dump(self.data, handle, -1)


class Metric():
    def __init__(self, data, underlying=None, name=None):
        self.data = data
        self.underlying = underlying
        self.name = name
        self.series = pd.Series(name=self.name)
        self.thresholds = {'Week': 7, 'Month': 30, 'Quarter': 91}

        self.last_2Q_data = self.data[
            self.data.created_at > (NOW - pd.Timedelta(days=184))
        ]
        self.time_frame = self.overTime(self.last_2Q_data).reset_index()
        breakpoint()
        self.setTimeComponents(self.time_frame, 'date')
        self.setTimeComponents(self.last_2Q_data, 'created_at', prefix=True)
        self.fillSeries()

    def overTime(self, data):
        dr = pd.DataFrame({
            'key': -1,
            'date': pd.date_range(START_DATETIME, NOW, freq='D')
        })
        data['key'] = -1
        merged = pd.merge(dr, data, on='key')
        data.drop('key', 1, inplace=True)
        merged['count'] = np.where(
            (merged.date > merged.created_at) & (
                (merged.date < merged.closed_at) | pd.isnull(merged.closed_at)
            ),
            True, False
        )

        cntOverTime = merged[merged['count']].groupby('date')['count'].count()
        cntOverTime = pd.concat(
            [dr.set_index('date'), cntOverTime], 1).drop('key', 1).fillna(0)
        return cntOverTime

    @staticmethod
    def setTimeComponents(frame, date_col, prefix=None):
        if prefix is None:
            prefix = ''
        elif prefix is True:
            prefix = f'{date_col}.'
        frame[prefix + 'Week'] = frame[date_col].dt.week
        frame[prefix + 'Month'] = frame[date_col].dt.month
        frame[prefix + 'Quarter'] = frame[date_col].dt.quarter

    def fillSeries(self):
        # 'Count', 'Proportion', 'Last_30_day_average', 'This_Week_average',
        # 'WeekTD', 'Prev_WeekTD', 'Week_dynamics', 'This_Month_average',
        # 'MonthTD', 'Prev_MonthTD', 'Month_dynamics', This_Quarter_average',
        # 'QuarterTD', 'Prev_QuarterTD', 'Quarter_dynamics'
        self.setCount()
        self.setProportion()
        self.setLast30DayAverage()
        self.setPeriodicData()

    def setCount(self):
        self.series['Count'] = len(self)

    def __len__(self):
        # if self.derived_from == 'issue':
        #     return len(self.open_data)
        return len(self.data)

    def setProportion(self):
        self.series['Proportion'] = self.ratio(True)

    def ratio(self, stringify=False):
        if self.underlying is not None:
            r = len(self) / len(self.underlying)
            return f'{r:.1%}' if stringify else r
        return np.nan

    # def setLast30Days(self):
    #     self.series['Last_30_day_average'] = self.time_frame.tail(30)[
    #         'count'].mean()
    def setLast30DayAverage(self):
        self.series['Last_30_day_average'] = self.time_frame[
            self.time_frame.date > (NOW - pd.Timedelta(days=30))
        ]['count'].mean()

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

            self.series[f'{period}TD'] = self.calcPeriodToDate(
                self.last_2Q_data,
                f'created_at.{period}', curWMQ[period]
            )

            self.series[f'Prev_{period}TD'] = self.calcPeriodToDate(
                self.last_2Q_data,
                f'created_at.{period}', lastWMQ[period]
            )
            self.setPeriodDynamics(period)

    def calcPeriodToDate(self, data, date_col, period_metric):
        return np.int64(len(data[data[date_col] == period_metric]))

    def setPeriodDynamics(self, period):
        self.series[f'{period}_dynamics'] = self._period_dynamics(
            self.series[f'{period}TD'], self.series[f'Prev_{period}TD']
        )

    def _period_dynamics(self, current, last):
        try:
            return (current - last) / last
        except ZeroDivisionError:
            if current > last:
                return float('inf')
            return np.nan


class CountMetric(Metric):
    pass


class TimeMetric(Metric):
    def fillSeries(self):
        pass


class MetricTable():

    def __init__(self, metrics):
        self.frame = pd.DataFrame([m.series for m in metrics])


class Dashboard():
    '''streamlit representation of the dashboard'''

    def __init__(self, metrics):
        self.time_start = time.time()
        self.metrics = metrics

    def draw(self):
        st.title('Analytics Dashboard')
        st.write('The metrics table', self.metrics)
        st.write('Dashboard load time:', time.time() - self.time_start)


if __name__ == '__main__':
    cached_raw_issues = GithubData('issues').getData({'state': 'all'})
    issueD = IssueData(
        cached_raw_issues
    )
    issueDTime = IssueData(cached_raw_issues)
    commitD = CommitData(
        GithubData('commits').getData()
    )
    labeledIssueD = issueD.detach(by=['state', 'label'], state='open')
    table = MetricTable([
        Metric(commitD.detach(), commitD, '(%) commits from the community'),
        *[Metric(
            issue, issueD, f'# of open {name}s'
        ) for name, issue in labeledIssueD.items()]
    ]).frame

    dashb = Dashboard(table)
    dashb.draw()
