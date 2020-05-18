import pandas                as pd
import numpy                 as np
import yfinance              as yf
from   scipy.signal          import savgol_filter
from   tqdm                  import tqdm
from   time                  import sleep, strftime, strptime, time
from   datetime              import timedelta, datetime, date
from   logging               import info, basicConfig, DEBUG, INFO
from   symbols               import STOCKS_FNM, EXCLUDE_FNM, EXCLUDE_SET, \
                                    LOGPATH


################################
### Logging helper functions ###
################################

def open_logfile(logpath, fnm):
    basicConfig(
        filename=f'{logpath}{fnm}', 
        format='%(asctime)s %(message)s', 
        level=INFO)

def log(msg, both=False):
    info(msg)
    if both == True:
        print(msg)

def log_filename(prefix):
    fnm = prefix + get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, fnm)

#############################
### Date helper functions ###
#############################

def get_current_day_and_time(fmt='%Y%m%d%H%M'):
    return strftime(fmt)

def get_current_day(fmt='%Y%m%d'):
    return strftime(fmt)

def add_days_to_date(ds, days, fmt='%Y-%m-%d'):
    ds_date = date.fromisoformat(str(ds)[:10])
    ds_date = ds_date + timedelta(days=int(days))
    return ds_date.__format__(fmt)

def is_leap_year(year, month, day):
    leap_year  = (year % 4) == 0 and (year % 100) != 0
    leap_year |= (year % 100) == 0 and (year % 400) == 0
    return leap_year

#                Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
days_in_month = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def is_valid_date(year, month, day):

    if (month < 1 or month > 12) or (day < 1 or day > 31):
        return False
    if day > days_in_month[month-1]:
        return month==2 and day==29 and is_leap_year(year, month, day)
    return True

def format_date(year, month, day):
    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    return date_str

def split_date(date_str):
    assert len(date_str) == 10, "date must be 10 chars"
    for i, c in enumerate(date_str):
        if i==4 or i==7:
            assert c=='-' or c=='/', "not properly separated date"
        else:
            assert c in "0123456789", "not a digit"
    
    y = int(date_str[0:4])
    m = int(date_str[5:7])
    d = int(date_str[8:10])

    if is_valid_date(y, m, d) == True:
        return y, m, d

    return -1, -1, -1

def add_years(date_str, to_add):
    return add_months(date_str, int(to_add * 12) )

def add_neg_months(y, m, d, to_add):
    y_to_subtract = int(abs(to_add)/12)
    m_to_subtract = abs(to_add) % 12

    y = y - y_to_subtract
    m = m - m_to_subtract

    if m <= 0:
        m = m + 12
        y = y - 1
    
    if d > days_in_month[m-1]:
        if m==2 and is_leap_year(y, m, d):
            m = m - 1
            d = d - (days_in_month[m] + 1)
        else:
            m = m - 1
            d = d - days_in_month[m]

    return y, m, d

def add_pos_months(y, m, d, to_add):
    y_to_add = int(to_add/12)
    m_to_add = to_add % 12

    y = y + y_to_add
    m = m - m_to_add

    if m >= 12:
        y = y + 1
        m = m - 12

    if d > days_in_month[m-1]:
        if m==2 and is_leap_year(y, m, d):
            m = m + 1
            d = d - (days_in_month[m] + 1)
        else:
            m = m + 1
            d = d - days_in_month[m]

    return y, m, d

def add_months(date_str, to_add):
    y, m, d = split_date(date_str)
    if is_valid_date(y, m, d) == False:
        return "invalid date"

    if to_add > 0:
        y, m, d = add_pos_months(y, m, d, to_add)
    else:
        y, m, d = add_neg_months(y, m, d, to_add)

    assert y > 1900 and y < 2100, "year outside of normal range"

    if is_valid_date(y, m, d) == False:
       assert m == 2 and d == 29, "must be 29-Feb!"
       d = d -1
    
    return format_date(y, m, d)


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

##########################################
### Calculate runtime helper functions ###
##########################################

def get_starttime():
    return time()

def calc_runtime(start_time, verbose):
    seconds = int(time() - start_time)
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    hours = int(minutes / 60)
    minutes = minutes - hours * 60
    log(f'Run time (hh:mm:ss) : {hours:02d}:{minutes:02d}:{seconds:02d}', verbose)

#############################################
### Get stock and smooth helper functions ###
#############################################

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

############################################
### Print ticker header helper functions ###
############################################

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

###############################
### Ticker helper functions ###
###############################

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

###############################
### Pandas helper functions ###
###############################

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

###############################
### String helper functions ###
###############################

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

def check_for_nulls(df):
    msg = ""
    nulls_exist = False
    cols = list(df.columns)
    for c in cols:
        has_null = any(df[c].isna())
        if has_null == True:
            msg = msg + f"Column {c} has nulls\n"
            nulls_exist = True
    msg = msg + "Finished checking...\n"
    return nulls_exist, msg

if __name__ == "__main__":

    y, m, d = split_date("2020-02-29")
    print(is_valid_date(y, m, d))

    y, m, d = split_date("2019-02-29")
    print(is_valid_date(y, m, d))

    print(add_months("2021-04-30", -14))
