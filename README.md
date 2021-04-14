# Analytics Dashboard for Activeloop

## Use
Deployed with Streamlit sharing. **The stable release is accessible [here](https://share.streamlit.io/haiyangdeperci/analytics-dashboard/dashboard_python37.py).**

You may also clone to your current directory and run the code locally in the following way:
```
git clone https://github.com/haiyangdeperci/analytics-dashboard.git .
pip install -r requirements.txt
streamlit run dashboard.py
```

## Notes
1. Streamlit sharing enforces Python 3.7 on the code, which leads to issues like *SyntaxErrors* in the original dashboard source file, `dashboard.py`. To stay compliant with Python 3.7, I use `python-walrus` package to modify the source with the following command:
    ```
    walrus -vs 3.8 -s dashboard.py > dashboard_python37.py
    ```
    The resulting file, `dashboard_python37.py`, is used in Streamlit sharing. You can adapt it for previous Python 3 versions in a similar fashion.

2. The requirements file for this code is generated with `pipreqs`:
    ```
    pipreqs --savepath requirements.txt
    ```
3. This code could be easily adapted to other repositories than `activeloopai/Hub` if needed.
4. The data is fetched and updated on refresh of the dashboard, however not more frequently than every hour.
