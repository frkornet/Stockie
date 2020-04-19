import pandas as pd
import numpy  as np
import gc; gc.collect()

from tqdm                    import tqdm
from util                    import print_ticker_heading, get_stock_n_smooth, smooth, log, \
                                    get_current_day_and_time, open_logfile, \
                                    is_holiday, exclude_tickers, build_ticker_list
from scipy.signal            import argrelmin, argrelmax

from sklearn.linear_model    import LogisticRegression
from category_encoders       import WOEEncoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline        import make_pipeline, Pipeline
from sklearn.preprocessing   import KBinsDiscretizer, FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.impute          import SimpleImputer

from symbols                 import BUY, SELL, STOCKS_FNM, EXCLUDE_FNM, \
                                    TRADE_FNM, BUY_FNM, LOGPATH, EXCLUDE_SET, \
                                    TRADE_PERIOD, TRADE_THRESHOLD, \
                                    TRADE_DAILY_RET

import warnings
warnings.filterwarnings("ignore")

def features(data, hist, target):
    """
    Given a standard yfinance data dataframe, add features that will help
    the balanced scorecard to recognize buy and sell signals in the data.
    The features are added as columns in the data dataframe. 
    
    The original hist dataframe from yfinance is provided, so we can copy
    the target to the data dataframe. The data dataframe with the extra 
    features is returned. The target argument contains the name of the 
    column that contains the the target.
    """
    windows = [3, 5, 10, 15, 20, 30, 45, 60]

    for i in windows:
        ma = data.Close.rolling(i).mean()
        # Moving Average Convergence Divergence (MACD)
        data['MACD_'+str(i)] = ma - data.Close
        data['PctDiff_'+str(i)] = data.Close.diff(i)
        data['StdDev_'+str(i)] = data.Close.rolling(i).std()

    factor = data.Close.copy()
    for c in data.columns.tolist():
        data[c] = data[c] / factor

    data[target] = hist[target]
    data = data.dropna()
    del data['Close']
    
    return data

def stringify(data):
    """
    Convert a Pandas dataframe with numeric columns into a dataframe with only
    columns of the data type string (in Pandas terminology an object). The 
    modified dataframe is returned. Note that the original dataframe is lost.
    """
    df = pd.DataFrame(data)
    for c in df.columns.tolist():
        df[c] = df[c].astype(str)
    return df

def split_data(stock_df, used_cols, target, train_pct):
    """
    Split data set into a training and test data set:
    - X contains the features for training and predicting. 
    - y contains the target for training and evaluating the performance.

    Used_cols contain the features (i.e. columns) that yiu want to use
    for training and prediction. Target contains the name of the column
    that is the target.

    Function returns X and y for cross validation, X and y for training, 
    and X and y for testing.
    """

    test_starts_at = int(len(stock_df)*train_pct)
    
    X = stock_df[used_cols]
    y = stock_df[target]

    X_train = stock_df[used_cols].iloc[:test_starts_at]
    X_test  = stock_df[used_cols].iloc[test_starts_at:]
    y_train = stock_df[target].iloc[:test_starts_at]
    y_test  = stock_df[target].iloc[test_starts_at:]

    return X, y, X_train, X_test, y_train, y_test

def get_signals(hist, target, threshold):
    """
    Used to predict buy and sell signals. The function itself has no awareness
    what it is predicting. It is just a helper function used by 
    get_possible_trades().

    Target is the column that contains the target. The other columns are
    considered to be features to be used for training and prection.

    The function uses a balanced weight of evidence scorecard to predict the 
    signals. It returns the signals array.

    Note that the function uses 70% for training and 30% for testing. The 
    date where the split happens is dependent on how much data the hist
    dataframe contains. So, the caller will not see a single split date for
    all tickers. 
    """
    # NB: we do not include smooth in data!
    data = hist[['Close', 'Open', 'Low', 'High']]
    data = features(data, hist, target)

    used_cols = [c for c in data.columns.tolist() if c not in [target]]
    X, y, X_train, X_test, y_train, y_test = split_data(data, used_cols, target, 0.7)

    encoder   = WOEEncoder()
    binner    = KBinsDiscretizer(n_bins=5, encode='ordinal')
    objectify = FunctionTransformer(func=stringify, check_inverse=False, validate=False)
    imputer   = SimpleImputer(strategy='constant', fill_value=0.0)
    clf       = LogisticRegression(class_weight='balanced', random_state=42)

    pipe = make_pipeline(binner, objectify, encoder, imputer, clf)
    pipe.fit(X_train, y_train.values)

    signals = (pipe.predict_proba(X_test)  > threshold).astype(int)[:,1]
    return signals

def merge_buy_n_sell_signals(buy_signals, sell_signals):
    """
    The functions will take two lists and produce a single containing the 
    buy and sell signals. The merged list will always start with a buy signal.
    This is achieved by setting the state to SELL. That ensures that all sell 
    signals are quietly dropped until we get to the first buy signal.

    Note: this function does not enforce that each buy signal is matched with  
    a sell signal.

    The function implements a simple deterministic state machine that flips 
    from SELL to BUY and back from BUY to SELL.

    A buy in the merged list is 1 and a sell is 2. The merged list is
    returned to the caller at the end.
    """

    assert len(buy_signals) == len(sell_signals), "buy_signal and sell_signal lengths different!"
    
    buy_n_sell = [0] * len(buy_signals)
    length     = len(buy_n_sell)
    i          = 0
    state      = SELL
    
    while i < length:
        if state == SELL and buy_signals[i] == 1:
            state = BUY
            buy_n_sell[i] = 1
        
        elif state == BUY and sell_signals[i] == 1:
            state = SELL
            buy_n_sell[i] = 2
            #continue
        
        i = i + 1
    
    return buy_n_sell

def extract_trades(hist, buy_n_sell, ticker, verbose):
    """
    Given a merged buy and sell list, extract all complete buy and sell pairs 
    and store each pair as a trade in a dataframe (i.e. possible_trades_df). 
    
    The possible trades dataframe contains the ticker, buy date, sell date, 
    the close price at buy date, the close price at sell data, the gain 
    percentage, and the daily compounded return.

    Note that hist, contains the data from yfinance for the ticker, so we
    can calculate the above values to be stored in the possible trades
    dataframe.

    The function returns the possible trades dataframe for a single ticker
    to the caller.
    
    The function assumes that the buy_n_sell list is well formed and does 
    not carry out any checks. Since the list is typiclly created by 
    merge_buy_n_sell_signals(), this should be the case.

    TODO: extend the functionality so that the buy at the end without 
    a matching signal is storted in an open position dataframe. The caller
    is then responsible for merging all open position of all tickers into 
    a single dataframe.  
    """

    test_start_at = len(hist) - len(buy_n_sell)
    
    cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
            'trading_days', 'daily_return', 'ticker' ]
    possible_trades_df = pd.DataFrame(columns=cols)
    
    buy_id = -1

    for i, b_or_s in enumerate(buy_n_sell):
        
        if b_or_s == BUY:
            buy_id    = test_start_at + i
            buy_close = hist.Close.iloc[buy_id]
            buy_date  = hist.index[buy_id]
            
        if b_or_s == SELL:
            sell_id    = test_start_at + i
            sell_close = hist.Close.iloc[sell_id]
            sell_date  = hist.index[sell_id] 
            
            gain = sell_close - buy_close
            gain_pct = round( (gain / buy_close)*100, 2)
            
            trading_days = sell_id - buy_id
            
            daily_return = (1+gain_pct/100) ** (1/trading_days) - 1
            daily_return = round(daily_return * 100, 2)
            
            trade_dict = {'buy_date'    : [buy_date],  'buy_close'    : [buy_close],
                         'sell_date'    : [sell_date], 'sell_close'   : [sell_close],
                         'gain_pct'     : [gain_pct],  'trading_days' : [trading_days],
                         'daily_return' : [daily_return], 'ticker'    : [ticker] }
            possible_trades_df = pd.concat([possible_trades_df, 
                                           pd.DataFrame(trade_dict)])
    
    if verbose == True:
        log("****EXTRACT_TRADES****")
        log(possible_trades_df)
    
    if buy_id > 0:
        buy_opportunity_df = {'ticker'    : [ticker] , 
                              'buy_date'  : [buy_date],  
                              'buy_close' : [buy_close],
                             }
        buy_opportunity_df = pd.DataFrame(buy_opportunity_df)
    else:
        cols=['ticker', 'buy_date', 'buy_close']
        buy_opportunity_df = pd.DataFrame(columns=cols)

    return possible_trades_df, buy_opportunity_df

def ticker_trades(ticker, verbose):
    target = 'target'
    gc.collect()
    if verbose == True:
        print_ticker_heading(ticker)

    success, hist = get_stock_n_smooth(ticker, TRADE_PERIOD)
    if success == False:
        return False, None, None

    try:
        # get the buy signals
        step = "get buy signals"
        hist[target] = 0
        min_ids = argrelmin(hist.smooth.values)[0].tolist()
        hist[target].iloc[min_ids] = 1        
        buy_signals = get_signals(hist, target, TRADE_THRESHOLD)

        # get the sell signals
        step = "get sell signals"
        hist[target] = 0
        max_ids = argrelmax(hist.smooth.values)[0].tolist()
        hist[target].iloc[max_ids] = 1
        sell_signals = get_signals(hist, target, TRADE_THRESHOLD)
        
        # merge the buy and sell signals
        step = "merge buy and sell signals"
        buy_n_sell = merge_buy_n_sell_signals(buy_signals, sell_signals)
        
        # extract trades
        step = "extract trades"
        ticker_df, buy_df = extract_trades(hist, buy_n_sell, ticker, verbose)
        return True, ticker_df, buy_df

    except:
        log(f"Failed to get possible trades for {ticker}")
        log(f"step={step}")
        return False, None, None
     
def get_possible_trades(tickers, threshold, period, verbose):
    """
    The main driver that calls other functions to do the work with the aim
    of extracting all possible trades for all tickers. For each ticker it
    performs the following steps:

    1) retrieve the historical ticker information (using yfinance),
    2) smooth the Close price curve,
    3) get the buy signals (using balanced scorecard),
    4) get the sell signals (using balanced scorecard),
    5) merge the buy and sell signals, and
    6) extract the possible trades and add that to the overall dataframe
       containing all possible trades for all tickers.

    The dataframe with all possible trades is then returned to the caller 
    at the end.
    """
    # print("tickers=", tickers)
    target = 'target'
    
    cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
            'trading_days', 'daily_return', 'ticker' ]
    possible_trades_df = pd.DataFrame(columns=cols)

    cols=['ticker', 'buy_date', 'buy_close']
    buy_opportunities_df = pd.DataFrame(columns=cols)
    
    #print('Determining possible trades...\n')
    for ticker in tqdm(tickers, desc="possible trades: "):
        success, ticker_df, buy_df = ticker_trades(ticker, verbose)
        if success == True:
            possible_trades_df = pd.concat([possible_trades_df, ticker_df])
            buy_opportunities_df = pd.concat([buy_opportunities_df, buy_df])

    possible_trades_df.trading_days = possible_trades_df.trading_days.astype(int)
    return possible_trades_df, buy_opportunities_df

def call_get_possible_trades(tickers):
    log("Calling get_possible_trades...")
    possible_trades_df, buy_opportunities_df \
       = get_possible_trades(tickers, 0.5, TRADE_PERIOD, False)
    log("Finished get_possible_trades")
    log("")
    return possible_trades_df, buy_opportunities_df

def drop_suspicious_trades(possible_trades_df):
    log("Checking for suspicious daily return trades...")
    idx = possible_trades_df.daily_return > TRADE_DAILY_RET
    log(f'possible_trades_df=\n{possible_trades_df[idx]}')
    if len(possible_trades_df[idx]) > 0:
        log(f'Dropping trades that exceed {TRADE_DAILY_RET}% daily return...')
    possible_trades_df = possible_trades_df[~idx]
    log('')
    return possible_trades_df


def save_files(possible_trades_df, buy_opportunities_df):
    log(f"Saving possible trades to {TRADE_FNM} ({len(possible_trades_df)})")
    possible_trades_df.to_csv(TRADE_FNM, index=False)
    log(f"Saving buy opportunities to {BUY_FNM} ({len(buy_opportunities_df)})")
    buy_opportunities_df.to_csv(BUY_FNM, index=False)
    log('')

def main():
    log("Generating possible trades")
    log('')

    tickers = build_ticker_list()
    possible_trades_df, buy_opportunities_df = call_get_possible_trades(tickers)
    possible_trades_df = drop_suspicious_trades(possible_trades_df)
    save_files(possible_trades_df, buy_opportunities_df)
    log('Done.')

if __name__ == "__main__":

    log_fnm = "trade"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        main()