import pandas                as pd
import numpy                 as np
import yfinance              as yf
from   scipy.signal          import savgol_filter
from   tqdm                  import tqdm
from   time                  import sleep, strftime, strptime
from   datetime              import timedelta, datetime, date
from   logging               import info, basicConfig, DEBUG, INFO
from   symbols               import STOCKS_FNM, EXCLUDE_FNM, EXCLUDE_SET

def open_logfile(logpath, fnm):
    basicConfig(
        filename=f'{logpath}{fnm}', 
        format='%(asctime)s %(message)s', 
        level=INFO)

def log(msg, both=False):
    info(msg)
    if both == True:
        print(msg)

def get_current_day_and_time(fmt='%Y%m%d%H%M'):
    return strftime(fmt)

def get_current_day(fmt='%Y%m%d'):
    return strftime(fmt)

def add_days_to_date(ds, days, fmt='%Y-%m-%d'):
    ds_date = date.fromisoformat(str(ds)[:10])
    ds_date = ds_date + timedelta(days=int(days))
    return ds_date.__format__(fmt)

Market_Holidays = ['20200101', '20200120', '20200217', '20200410', '20200525', 
                   '20200703', '20200907', '20201126', '20201225',

                   '20210101', '20210118', '20210215', '20210402', '20210531',
                   '20210705', '20210906', '20211125', '20211224',
                   
                   '20220117', '20220221', '20220415', '20220530',
                   '20220704', '20220905', '20221124', '20221226',

                   '20230102', '20230116', '20230220', '20230407', '20230529',
                   '20230704', '20230904', '20231123', '20231225'
                  ]

def is_holiday():
    today = get_current_day()
    return today in Market_Holidays

def smooth(hist, ticker):
    """
    Smooth the Close price curve of hist data frame returned by yfinance. Two
    values are returned. The first is whether or not the smooth was successful 
    (True is successful and False is unsuccessful). The second value is the
    hist dataframe with an extra column smooth containing the smoothed Close
    curve.
    """
    window = 15

    try:
        #print(ticker)
        hist['smooth'] = savgol_filter(hist.Close, 2*window+1, polyorder=3)
        hist['smooth'] = savgol_filter(hist.smooth, 2*window+1, polyorder=1)
        hist['smooth'] = savgol_filter(hist.smooth, window, polyorder=3)   
        hist['smooth'] = savgol_filter(hist.smooth, window, polyorder=1)
        return True, hist
    except:
        #print(f"Failed to smooth prices for {ticker}!")
        return False, hist

def get_stock_period(ticker, times, period):
    """
    Try up to `times` to get data for ticker from yfinance. After
    each failure sleep for 5 seconds before trying again.
    """
    for i in range(times):
        try:
            asset  = yf.Ticker(ticker)
            hist   = asset.history(period=period)
            return True, hist
        except:
            print(f"Failed to retrieve stock data for {ticker} (attempt={i+1})")
            sleep(5)
            continue
    
    hist=pd.DataFrame(columns=['Open', 'Close', 'High', 'Low'])
    return False, hist

def get_stock_n_smooth(ticker, period):
    success, hist = get_stock_period(ticker, 2, period)
    if success == False:
        return False, hist
    return smooth(hist, ticker)

def get_stock_start(ticker, times, start, end):
    """
    Try up to `times` to get data for ticker from yfinance. After
    each failure sleep for 5 seconds before trying again.
    """
    for i in range(times):
        try:
            asset  = yf.Ticker(ticker)
            hist   = asset.history(start=start, end=end)
            return True, hist
        except:
            print(f"Failed to retrieve stock data for {ticker} (attempt={i+1})")
            sleep(5)
            continue
    
    hist=pd.DataFrame(columns=['Open', 'Close', 'High', 'Low'])
    return False, hist

def print_ticker_heading(ticker):
    """
    Print a heading for the ticker on the console. Nothing is returned. Is
    a helper function only to avoid having to duplicate the same code over
    and over again. 
    """
    print("*******************")
    print("***", ticker, " "*6, "***" )
    print("*******************")
    print('')

def exclude_tickers(tickers, exclude_list):
    t_list = []
    for t in tickers:
        if t not in exclude_list:
            t_list.append(t)        
    return sorted(t_list)

def build_ticker_list():
    log(f'Reading stocks from {STOCKS_FNM}')
    sdf = pd.read_csv(STOCKS_FNM)
    log(f'len(sdf)={len(sdf)}')

    log(f'Reading exclude list from {EXCLUDE_FNM}')
    edf = pd.read_csv(EXCLUDE_FNM)
    log(f'len(edf)={len(edf)}')

    log("Eliminate tickers that are on the exclude list")
    tickers = exclude_tickers(sdf.TICKER.to_list(), edf.ticker.to_list())
    log(f"len(tickers)={len(tickers)}")
    log("")

    log("Eliminate tickers that are on the hard-coded exclude list")
    log(f'EXCLUDE_SET={EXCLUDE_SET}')
    tickers = exclude_tickers(tickers, EXCLUDE_SET)
    log(f"len(tickers)={len(tickers)}")
    log("")

    return tickers

def empty_dataframe(cols, data_types):
    assert len(cols) == len(data_types), "len(cols) <> len(data_types)"

    df = pd.DataFrame(columns=cols)
    for i, c in enumerate(cols):
        dt = data_types[i]
        df[c] = df[c].astype(dt)

    return df

def dict_to_dataframe(df_dict, data_types):

    df = pd.DataFrame(df_dict)
    assert len(df.columns)==len(data_types), "len(df.cols) <> len(data_types)"

    for i, c in enumerate(df.columns):
        dt = data_types[i]
        df[c] = df[c].astype(dt)

    return df

def save_dataframe_to_csv(df, fnm):
    df.to_csv(fnm, index=False)

def trim_tickers(str):
    if len(str) == 0:
        return ''
    str = str.replace('[', '')
    str = str.replace(']', '')
    str = str.replace(' ', '')
    str = str.replace('\'', '')
    return str

def add_spaces(s: str, width):
    s_len = len(s)
    spaces_to_add = width - s_len
    if spaces_to_add <= 0:
        return s
    s = s + " " * spaces_to_add
    return s

def set_to_string(my_set, width):
    my_set = sorted(my_set)
    set_str = str(my_set)
    set_str = trim_tickers(set_str)
    set_str = add_spaces(set_str, width)
    return str(set_str[:width])