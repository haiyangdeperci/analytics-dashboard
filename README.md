# Analytics Dashboard for Activeloop

## Use
Deployed with Streamlit sharing. **You can access it directly by clicking [here](https://share.streamlit.io/haiyangdeperci/analytics-dashboard/hotfix/data/dashboard_python37.py)**. The stable release is <s>accessible [here](https://share.streamlit.io/haiyangdeperci/analytics-dashboard/dashboard_python37.py)</s> not ready yet.

You may also clone to your current directory and run the code locally in the following way:
```
git clone https://github.com/haiyangdeperci/analytics-dashboard.git .
pip install -r requirements.txt
streamlit run dashboard.py
```

## Notes
1. Streamlit sharing enforces Python 3.7 on the code, which leads to issues like SyntaxErrors in the original dashboard source file, `dashboard.py`. To stay compliant with Python 3.7, I use `python-walrus` package to modify the source with the following command:
    ```
    walrus -vs 3.8 -s dashboard.py > dashboard_python37.py
    ```
    The resulting file, `dashboard_python37.py`, is used in Streamlit sharing.

2. This code could be easily adapted to other repositories than `activeloopai/Hub` if needed.
3. As part of a temporary solution, the data is fetched from Github and updated periodically.
