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
