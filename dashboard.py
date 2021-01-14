import streamlit as st
import requests
import os
import numpy as np
import pandas as pd

GITHUB_API_REPO_URL = 'https://api.github.com/repos/activeloopai/Hub/'
GITHUB_API_ORG_URL = 'https://api.github.com/orgs/activeloopai/'
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')


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


open_bugs = getIssues('open', 'bug')
st.write("Number of open bugs:", len(open_bugs))

open_issues = getIssues('open')
open_good_first_issues = getIssues('open', 'good first issue')
st.write(
    "Number of good first issues / all issues:",
    f'{len(open_good_first_issues)} / {len(open_issues)} '
    f'({len(open_good_first_issues) / len(open_issues):.1%})',
)
