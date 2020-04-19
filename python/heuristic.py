import pandas as pd
import numpy as np
from symbols import TRADE_FNM, STATS_FNM, LOGPATH
from util    import open_logfile, log, is_holiday, get_current_day_and_time
from tqdm    import tqdm

import matplotlib.pyplot as plt

import gc; gc.enable()

def read_possible_trades():
    possible_trades_df = pd.read_csv(TRADE_FNM)
    return possible_trades_df

def print_possible_trades_stats(trades_df):
    trading_days = trades_df['trading_days']
    gain_pct_df = trades_df['gain_pct']

    starting_at = int(len(trades_df)*0.7)
    train_trades_df = trades_df[:starting_at]
    test_trades_df = trades_df[:starting_at]

    fig, axs = plt.subplots(3)

    trades_df.plot.scatter(x='trading_days', y='gain_pct', ax=axs[0])
    axs[0].set_title('Possible trades: trading days vs percentage gain (full set)')

    train_trades_df.plot.scatter(x='trading_days', y='gain_pct', ax=axs[1])
    axs[1].set_title('Possible trades: trading days vs percentage gain (training set)')
        
    test_trades_df.plot.scatter(x='trading_days', y='gain_pct', ax=axs[2])
    axs[2].set_title('Possible trades: trading days vs percentage gain (test set)')

    plt.show()

def calc_ticker_stats(ticker_df):
    cnt_gain  = len(ticker_df)
    if cnt_gain > 0:
        min_gain  = float(ticker_df.gain.min())
        idx = ticker_df.gain == min_gain
        assert len(ticker_df.loc[idx]) > 0, "Unable to find minimum gains!"
        mean_buy_close = ticker_df[idx].buy_close.mean()
        min_pct = float(min_gain  / mean_buy_close) * 100

        max_gain  = float(ticker_df.gain.max())
        idx = ticker_df.gain == max_gain
        assert len(ticker_df.loc[idx]) > 0, "Unable to find maximum gains!"
        mean_buy_close = ticker_df[idx].buy_close.mean()
        max_pct = float(max_gain  / mean_buy_close) * 100

        mean_gain = float(ticker_df.gain.mean())
        std_gain  = float(ticker_df.gain.std())
        day_gain  = float(ticker_df.trading_days.mean())
        gain_sum  = float(ticker_df.buy_close.sum()) 
        pct_gain  = ( (mean_gain * cnt_gain) / gain_sum ) * 100
        gain_daily_return = (1+pct_gain/100) ** (1/(day_gain * cnt_gain)) - 1
        gain_daily_return = gain_daily_return * 100.0
    else:
        min_gain  = 0.0
        min_pct   = 0.0
        max_gain  = 0.0
        max_pct   = 0.0
        mean_gain = 0.0
        std_gain  = 0.0
        day_gain  = 0.0
        gain_sum  = 0.0
        pct_gain  = 0.0
        gain_daily_return = 0.0

    tuple = (cnt_gain, min_gain, min_pct, max_gain, max_pct,
             mean_gain, std_gain, day_gain, gain_sum, 
             pct_gain, gain_daily_return)
    return tuple

def calc_zero_stats(ticker_df):
    zero_cnt   = len(ticker_df)
    if zero_cnt > 0:
        day_zero = float(ticker_df.trading_days.mean())
    else:
        day_zero = 0.0
    return zero_cnt, day_zero

def calc_total_stats(ticker_df):
    # Stop-loss kicks in at -10% => cap losses at 15% of buy_close
    # This is a hack that is skewing the statistics. Needs rework...
    # ticker_df.gain.loc[ticker_df.gain/ticker_df.buy_close < -0.1] = ticker_df.buy_close * (-0.15)
    
    total_buy  = float(ticker_df.buy_close.sum())
    total_sell = float(ticker_df.sell_close.sum())
    total_days = int(ticker_df.trading_days.sum())
    total_cnt  = int(ticker_df.trading_days.count())
    total_pct  = float(( (total_sell - total_buy) / total_buy ) * 100)
       
    expected_ret = 1 + total_pct/100
    if expected_ret > 0:
        daily_ret  = (1 + total_pct/100) ** (1/(total_days)) - 1
        daily_ret  = daily_ret * 100.0
    else:
        daily_ret = -np.Inf #  0.0

    tuple = (total_buy, total_sell, total_days, total_cnt, total_pct, expected_ret, daily_ret)
    return tuple

def calc_stats_ticker(trades_df, ticker):
    ticker_df = trades_df.loc[trades_df.ticker == ticker]
    gain_df = ticker_df.loc[ticker_df.gain > 0]
    loss_df = ticker_df.loc[ticker_df.gain < 0]
    zero_df = ticker_df.loc[ticker_df.gain == 0]

    cnt_gain, min_gain, min_pct_gain, max_gain, max_pct_gain, mean_gain,\
         std_gain, day_gain, gain_sum, pct_gain, gain_daily_return \
        = calc_ticker_stats(gain_df)
    cnt_loss, min_loss, min_pct_loss, max_loss, max_pct_loss, mean_loss, \
        std_loss, day_loss, loss_sum, pct_loss, loss_daily_return \
        = calc_ticker_stats(loss_df)

    zero_cnt, day_zero = calc_zero_stats(zero_df)
    total_buy, total_sell, total_days, total_cnt, total_pct, expected_ret, \
        daily_ret = calc_total_stats(ticker_df)

    # % of trades over 0.5 % per day compounded
    desired_cnt   = len(ticker_df[ticker_df.daily_return > 0.5])
    len_ticker_df = len(ticker_df)
    pct_desired   = (desired_cnt / len_ticker_df) * 100
    pct_desired   = round(pct_desired, 2)

    t_dict = {'ticker'     : [ticker], 
              'min_gain'   : [min_gain],   'min_pct_gain':  [min_pct_gain],
              'max_gain'   : [max_gain],   'max_pct_gain':  [max_pct_gain],
              'mean_gain'  : [mean_gain],  'std_gain': [std_gain], 
              'cnt_gain'   : [cnt_gain],   'pct_gain': [pct_gain],
              'day_gain'   : [day_gain],   'gain_daily_ret': [gain_daily_return],
              'min_loss'   : [min_loss],   'min_pct_loss': [min_pct_loss],
              'max_loss'   : [max_loss],   'max_pct_gain':  [max_pct_loss],
              'mean_loss'  : [mean_loss],  'std_loss': [std_loss], 
              'cnt_loss'   : [cnt_loss],   'pct_loss': [pct_loss],
              'day_loss'   : [day_loss],   'loss_daily_ret': [loss_daily_return],
              'day_zero'   : [day_zero],   'zero_cnt'  : [zero_cnt],
              'total_buy'  : [total_buy],  'total_sell': [total_sell], 
              'total_days' : [total_days], 'total_cnt' : [total_cnt],
              'total_pct'  : [total_pct],  'daily_ret' : [daily_ret],
              'pct_desired': [pct_desired]
             }
    log(f't_dict={t_dict}')

    return pd.DataFrame(t_dict)

def calc_stats(trades_df):
    trades_df['gain'] = trades_df.sell_close - trades_df.buy_close
    tickers = list(trades_df.ticker.unique())

    cols = ['ticker', 
            'min_gain',  'min_pct_gain',  'max_gain',  'max_pct_gain',
            'mean_gain', 'std_gain', 'cnt_gain',  'pct_gain', 'day_gain', 'gain_daily_ret',
            'min_loss',  'min_pct_loss', 'max_loss', 'max_pct_loss',
            'mean_loss', 'std_loss', 'cnt_loss',  'pct_loss', 'day_loss', 'loss_daily_ret',
            'day_zero',  'zero_cnt',  
            'total_buy', 'total_sell', 'total_days', 'total_cnt', 'total_pct', 'daily_ret',
            'pct_desired'
        ]
    ticker_stats_df = pd.DataFrame(columns=cols)

    for t in tqdm(tickers):
        gc.collect()
        ticker_stats_df = pd.concat( [ticker_stats_df, calc_stats_ticker(trades_df, t)], sort=False )
        #log('')
        #log(f"ticker_stats_df after adding {t}")
        #log(f'\n{ticker_stats_df}')

    ticker_stats_df.cnt_gain   = ticker_stats_df.cnt_gain.astype(int)
    ticker_stats_df.cnt_loss   = ticker_stats_df.cnt_loss.astype(int)
    ticker_stats_df.zero_cnt   = ticker_stats_df.zero_cnt.astype(int)
    ticker_stats_df.total_days = ticker_stats_df.total_days.astype(int)
    ticker_stats_df.total_cnt  = ticker_stats_df.total_cnt.astype(int)

    return ticker_stats_df

def post_processing(ticker_stats_df):
    # Make sure that column of type integer
    ticker_stats_df.cnt_gain   = ticker_stats_df.cnt_gain.astype(int)
    ticker_stats_df.cnt_loss   = ticker_stats_df.cnt_loss.astype(int)
    ticker_stats_df.zero_cnt   = ticker_stats_df.zero_cnt.astype(int)
    ticker_stats_df.total_days = ticker_stats_df.total_days.astype(int)
    ticker_stats_df.total_cnt  = ticker_stats_df.total_cnt.astype(int)

    # Calculate gain ratio
    ticker_stats_df['gain_ratio'] = ticker_stats_df.cnt_gain / (ticker_stats_df.total_cnt)

    # Calculate expected daily returns for gains and losses
    ticker_stats_df['e_gain_daily_ret'] = ticker_stats_df.gain_daily_ret * ticker_stats_df.gain_ratio
    ticker_stats_df['e_loss_daily_ret'] = ticker_stats_df.gain_daily_ret * (1-ticker_stats_df.gain_ratio)

    log("ticker_stats_df.describe=")
    log(f'\n{ticker_stats_df.describe()}')

    return ticker_stats_df

def heuristic(ticker_stats_df, trades_df):
    idx = (ticker_stats_df.daily_ret > 0) & (ticker_stats_df.gain_ratio > 0.7) \
        & (ticker_stats_df.pct_gain > abs(ticker_stats_df.pct_loss))
    good_tickers = list(set(ticker_stats_df.ticker.loc[idx].to_list()))
    log(f'len(good_tickers)={len(good_tickers)}')
    log(f'good_tickers={good_tickers}')

    # Build index that identifies the good tickers in trades_df
    idx = trades_df.ticker == ''
    for t in good_tickers:
        idx |= trades_df.ticker == t

    log(f'len(trades_df)={len(trades_df)}')
    log(f'len(trades_df[idx])={len(trades_df[idx])}')
    ratio = round(len(trades_df[idx])/len(trades_df), 2)
    log(f'ratio={ratio}')

    return good_tickers, idx

    # # cap losses at -15% (stop-loss)
    # temp_df.gain_pct.loc[temp_df.gain_pct < -15.0] = -15.0
    # temp_df    

def calc_n_print_ratios(trades_df):
    gain_sum, gain_cnt = trades_df['gain'][trades_df.gain >  0].agg(['sum', 'count'])
    loss_sum, loss_cnt = trades_df['gain'][trades_df.gain <  0].agg(['sum', 'count'])
    zero_sum, zero_cnt = trades_df['gain'][trades_df.gain == 0].agg(['sum', 'count'])

    log('')
    log(f'gain_sum={gain_sum} gain_cnt={gain_cnt}')
    log(f'loss_sum={loss_sum} loss_cnt={loss_cnt}')
    log(f'zero_sum={loss_sum} zero_cnt={loss_cnt}')
    gain_share = (gain_cnt / (gain_cnt + loss_cnt + zero_cnt)) * 100
    gain_share = round(gain_share,2)
    log(f'gain_share={gain_share} %')
    log('')

def save_good_tickers(ticker_stats_df, good_tickers):
    ticker_stats_df['good'] = 0
    for t in good_tickers:
        ticker_stats_df.good.loc[ticker_stats_df.ticker == t] = 1
    
    log('')
    log(f'len(good_tickers)={len(good_tickers)}')
    log(f'sum(ticker_stats_df.good)={sum(ticker_stats_df.good)}')

    ticker_stats_df.to_csv(STATS_FNM, index=False)    

def main():
    trades_df = read_possible_trades()
    #print_possible_trades_stats(trades_df)
    ticker_stats_df = calc_stats(trades_df)
    ticker_stats_df = post_processing(ticker_stats_df)
    good_tickers, idx = heuristic(ticker_stats_df, trades_df)
    print_possible_trades_stats(trades_df[idx])
    log('trades_df (all tickers):')
    calc_n_print_ratios(trades_df)
    log('trades_df.loc[idx] (good tickers):')
    calc_n_print_ratios(trades_df)
    save_good_tickers(ticker_stats_df, good_tickers)


if __name__ == "__main__":

    log_fnm = "stats"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        main()
