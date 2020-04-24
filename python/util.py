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

def get_stock(ticker, times, period):
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
    success, hist = get_stock(ticker, 2, period)
    if success == False:
        return False, hist
    return smooth(hist, ticker)

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


def empty_dataframe(cols):
    return pd.DataFrame(columns=cols)