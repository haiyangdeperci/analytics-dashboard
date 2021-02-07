import time
import streamlit as st
import requests
import os
import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import pickle

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
        self.dataDir = 'data'
        if not os.path.exists(self.dataDir):
            os.mkdir(self.dataDir)

    def retrievePage(self, params):
        RawData._api_calls += 1
        if RawData._api_calls > 60:
            raise TooManyAPICalls

        return requests.get(
            self.url, params=params,
            headers=self.headers
        ).json()

    def getData(self, params=None, *, single=False, load=True, save=True):
        if False:
            load = NotImplemented

        def __walrus_wrapper_load_8f6fb571cc224c3faa05234636cf4d53(expr):
            """Wrapper function for assignment expression."""
            nonlocal load
            load = expr
            return load

        if load and (__walrus_wrapper_load_8f6fb571cc224c3faa05234636cf4d53(self.store())):
            return load
        params = params or self.params
        params['per_page'] = 100
        params['page'] = 1
        if single:
            results = self.retrievePage(params)
        else:
            results = []

            if False:
                r = NotImplemented

            def __walrus_wrapper_r_a67ecb33a1c14ee49c863d7a1c59f2c2(expr):
                """Wrapper function for assignment expression."""
                nonlocal r
                r = expr
                return r

            while __walrus_wrapper_r_a67ecb33a1c14ee49c863d7a1c59f2c2(self.retrievePage(params)):
                params['page'] += 1
                results += r
        self.store(results)
        return results

    def store(self, results=None):
        if self.storage_file:
            path = os.path.join(self.dataDir, self.storage_file)
        if results is not None or os.path.exists(path):
            mode = 'wb' if results else 'rb'
            with open(path, mode) as handle:
                if mode == 'rb':
                    fileTimestamp = pd.to_datetime(os.path.getmtime(path), unit='s', utc=True)
                    if NOW < fileTimestamp + pd.Timedelta(hours=1):
                        return pickle.load(handle)
                else:
                    pickle.dump(results, handle, -1)


class GithubData(RawData):
    def __init__(self, resource, subresource=None, *, params=None, headers=None):
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
        self.headers = {
            'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
        headers = headers or {}
        self.headers = {**self.headers, **headers}
        self.base_url = 'https://api.github.com'
        self.org = 'activeloopai'
        self.repo = 'Hub'
        self.resource = resource
        self.subresource = subresource
        self.storage_file = f"{self.repo}_GH_{self.resource}.pkl"
        if self.subresource:
            self.storage_file += f"_{self.subresource}"
        self.divisions = {
            'issues': 'repos',
            'commits': 'repos',
            'traffic': 'repos',
            'stargazers': 'repos',
            'members': 'orgs',
        }
        self.params = params or {}
        super().__init__(self.build_url())

    def build_url(self):
        division = self.divisions[self.resource]
        url = f'{self.base_url}/{division}/{self.org}/'
        if division != 'orgs':
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

    def forMember(self, member):
        self.data = self.data[~self.data['is_community']]
        return self.data[self.data.created_by == member]


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
        self.dataDir = 'data'
        self.old_data = self.load()
        super().__init__(data)
        if self.old_data is not None:
            # this could be improved to get data only X days
            self.data = self.old_data.append(
                self.data, ignore_index=True).drop_duplicates('created_at', keep='last')
        if save:
            self.save()

    def parse(self):
        self.data = pd.json_normalize(self.data.views)
        self.data['timestamp'] = pd.to_datetime(self.data.timestamp, utc=True)
        self.data = self.data.rename(columns={'timestamp': 'created_at'})
        self.data['closed_at'] = pd.Series(pd.NaT, dtype='datetime64[ns, UTC]')

    def load(self):
        path = os.path.join(self.dataDir, self.storage_file)
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                return pickle.load(handle)

    def save(self):
        path = os.path.join(self.dataDir, self.storage_file)
        with open(path, 'wb') as handle:
            pickle.dump(self.data, handle, -1)


class StargazerData(Data):
    def parse(self):
        self.data[[
            'user_name', 'user_id'
        ]] = self.data.user.apply(self.parseNested, args=('login', 'id'))
        self.data.drop(['user'], 1, inplace=True)
        self.data = self.data.rename(columns={'starred_at': 'created_at'})
        self.data['created_at'] = pd.to_datetime(self.data.created_at, utc=True)
        self.data['closed_at'] = pd.Series(pd.NaT, dtype='datetime64[ns, UTC]')


class Metric():
    def __init__(self, data, underlying=None, name=None, measurements=None):
        self.data = data
        self.underlying = underlying
        self.name = name
        self.time_idx_name = 'created_at'
        self.thresholds = {'Week': 7, 'Month': 30, 'Quarter': 91}

        self.data = self.data.sort_values(
            self.time_idx_name).reset_index(drop=True)
        self.last_2Q_data = self.data[
            self.data[self.time_idx_name] > (NOW - pd.Timedelta(days=184))
        ]
        self.customSetUp(measurements)
        cols = self.last_2Q_data.columns
        self.last_2Q_data = self.last_2Q_data[
            cols.intersection(self.measurements + [self.time_idx_name])
        ]

        self.setTimeComponents(self.time_frame, self.tf_idx_name)
        self.setTimeComponents(
            self.last_2Q_data, self.time_idx_name, prefix=True)
        self.fillResult()

    def customSetUp(self):
        pass

    @staticmethod
    def setTimeComponents(frame, date_col, prefix=None):
        if prefix is None:
            prefix = ''
        elif prefix is True:
            prefix = f'{date_col}.'
        frame[prefix + 'Week'] = frame[date_col].dt.week
        frame[prefix + 'Month'] = frame[date_col].dt.month
        frame[prefix + 'Quarter'] = frame[date_col].dt.quarter

    def fillResult(self):
        # 'Count', 'Proportion', 'Last_30_day_average', 'This_Week_average',
        # 'WeekTD', 'Prev_WeekTD', 'Week_dynamics', 'This_Month_average',
        # 'MonthTD', 'Prev_MonthTD', 'Month_dynamics', This_Quarter_average',
        # 'QuarterTD', 'Prev_QuarterTD', 'Quarter_dynamics'
        self.setCount()
        self.setProportion()
        self.setLast30DayAverage()
        self.setPeriodicData()

    def setCount(self):
        self.result['Count'] = self._count(self)

    def __len__(self):
        # if self.derived_from == 'issue':
        #     return len(self.open_data)
        return len(self.data)

    def setProportion(self):
        self.result['Proportion'] = self.ratio(True)

    def ratio(self, stringify=False):
        if self.underlying is not None:
            r = len(self) / len(self.underlying)
            return f'{r:.1%}' if stringify else r
        return np.nan

    def setLast30DayAverage(self):
        self.result['Last_30_day_average'] = self.time_frame[
            self.time_frame[self.tf_idx_name] > (NOW - pd.Timedelta(days=30))
        ][self.measurements].mean().squeeze()

    def setPeriodicData(self):
        curWMQ = self.time_frame[
            list(self.thresholds)].iloc[-1]
        lastWMQ = pd.Series()

        for period in self.thresholds:
            last_idx = self.time_frame[
                self.time_frame[period] == curWMQ[period]
            ].first_valid_index() - 1
            if last_idx > 0:
                lastWMQ[period] = self.time_frame.loc[last_idx][period]
            else:
                lastWMQ[period] = np.nan

            self.result[f'This_{period}_average'] = self.time_frame[
                self.time_frame[period] == curWMQ[period]][
                    self.measurements].mean().squeeze()

            self.result[f'{period}TD'] = self.calcPeriodToDate(
                self.last_2Q_data,
                f'{self.time_idx_name}.{period}', curWMQ[period]
            )

            self.result[f'Prev_{period}TD'] = self.calcPeriodToDate(
                self.last_2Q_data,
                f'{self.time_idx_name}.{period}', lastWMQ[period]
            )
            self.setPeriodDynamics(period)

    def calcPeriodToDate(self, data, date_col, period_metric):
        return self._count(data[data[date_col] == period_metric])

    def _count(self, collection):
        return np.int64(len(collection))

    def setPeriodDynamics(self, period):
        self.result[f'{period}_dynamics'] = self._period_dynamics(
            self.result[f'{period}TD'], self.result[f'Prev_{period}TD']
        )

    def _period_dynamics(self, current, last):
        try:
            return (current - last) / last
        except ZeroDivisionError:
            if current > last:
                return float('inf')
            return np.nan

    def __getattr__(self, name):
        return getattr(self.data, name)


class CountMetric(Metric):
    def customSetUp(self, measurements):
        self.tf_idx_name = 'date'
        self.measurements = ['count']
        self.result = pd.Series(name=self.name)
        self.time_frame = self.overTime(self.last_2Q_data).reset_index()

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


class TimeMetric(Metric):
    def customSetUp(self, measurements):
        self.tf_idx_name = self.time_idx_name
        self.measurements = measurements or [
            'actual_resolution_time',
            'predicted_resolution_time'
        ]
        named_measurements = [f'{self.name}.{m}' for m in self.measurements]
        name_map = dict(zip(
            self.measurements, named_measurements
        ))
        self.measurements = named_measurements
        self.data = self.data.rename(columns=name_map)
        self.last_2Q_data = self.last_2Q_data.rename(columns=name_map)
        self.result = pd.DataFrame()
        self.time_frame = self.last_2Q_data

    def _count(self, collection):
        if isinstance(collection, TimeMetric):
            collection = self.data
        collection = collection[self.measurements]
        try:
            return collection.sum()
        except TypeError:
            return pd.Series(index=collection.columns, dtype='timedelta64[ns]')

    def __iter__(self):
        row_result = self.result.T
        return iter(row_result[s] for s in row_result)


class TeamFilter():
    def __init__(self, data):
        self.member_data = {}
        self.data = data

    def filter(self):
        for member in Data.whitelist():
            self.member_data[member] = self.data.forMember(member)

class MetricTable():

    def __init__(self, metrics):
        all_series = []
        for m in metrics:
            if isinstance(m, TimeMetric):
                all_series.extend([*m])
            else:
                all_series.append(m.result)
        self.frame = pd.DataFrame(all_series)


class Dashboard():
    '''streamlit representation of the dashboard'''

    def __init__(self):
        self.time_start = time.time()

    def receive(self, metrics):
        self.metrics = metrics

    def draw_canvas(self):
        st.title('Analytics Dashboard')

    def draw_metrics(self):
        st.write('The metrics table', self.metrics)
        st.write('Dashboard load time:', time.time() - self.time_start)


if __name__ == '__main__':
    load_dotenv()
    dashb = Dashboard()
    dashb.draw_canvas()
    cached_raw_visitors = GithubData('traffic', 'views').getData(single=True)
    cached_raw_issues = GithubData('issues').getData({'state': 'all'})
    cached_raw_comments = GithubData('issues', 'comments').getData()
    cached_raw_stargazers = GithubData(
        'stargazers',
        headers={'Accept': 'application/vnd.github.v3.star+json'}
    ).getData()

    stargazerD = StargazerData(cached_raw_stargazers)

    visitorD = VisitorData(cached_raw_visitors, save=True)
    uniqueVisitors = visitorD.drop('count', 1)
    allVisitors = visitorD.drop('uniques', 1)

    issueD = IssueData(cached_raw_issues, parse=False)
    issueD.parse(False, False)
    commentD = CommentData(cached_raw_comments, issueD)

    pullRequestD = IssueData(cached_raw_issues, parse=False)
    pullRequestD.parse(False, True)
    pullRequestD.detach(by='community')
    pullRequestD = pullRequestD.data

    bugD = IssueData(
        cached_raw_issues).detach(by=['label'], labels=['bug'])['bug']


    # eventually it should be just issueD for all
    # currently .detach() breaks the structure of the collection
    # this should be fixed
    issueDCount = IssueData(
        cached_raw_issues
    )
    issueDTime = IssueData(cached_raw_issues).data
    commitD = CommitData(
        GithubData('commits').getData()
    )
    # issueDCount loses unclosed issues
    labeledIssueD = issueDCount.detach(by=['state', 'label'], state='open')

    # test
    stargM = CountMetric(stargazerD, name='stars')
    uniqueVisM = TimeMetric(uniqueVisitors, name='visitors', measurements=['uniques'])
    starpUniqVis = stargM.time_frame.rename(columns={'date': 'created_at'}).merge(uniqueVisM)
    starpUniqVis['star/uVis'] = starpUniqVis['count'].diff() / starpUniqVis['visitors.uniques']

    table = MetricTable([
        CountMetric(
            commitD.detach(), commitD, '(%) commits from the community'
        ),
        *[CountMetric(
            issue, issueDCount, f'# of open {name}s'  # OK: out of open issues
        ) for name, issue in labeledIssueD.items()],
        TimeMetric(issueDTime, name='issue'),
        TimeMetric(pullRequestD, name='pr'),
        TimeMetric(bugD, name='bug'),
        TimeMetric(commentD, name='to_first_comment'),  # time to first comment
        TimeMetric(
            uniqueVisitors, name='visitors', measurements=['uniques']),
        TimeMetric(allVisitors, name='visitors', measurements=['count']),
        CountMetric(stargazerD, name='stars'),
        TimeMetric(starpUniqVis, name='', measurements=['star/uVis'])

    ]).frame

    dashb.receive(table)
    dashb.draw_metrics()
