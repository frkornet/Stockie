import pandas as pd
import numpy as np
from symbols import TRAIN_TRADE_FNM, TEST_TRADE_FNM, STATS_FNM, LOGPATH, \
                    PICPATH, STAT_COLS,STAT_COL_TYPES, TRADE_COLS, \
                    TRADE_COL_TYPES, STAT_BATCH_SIZE
from util    import open_logfile, log, is_holiday, get_current_day_and_time, \
                    empty_dataframe, dict_to_dataframe, save_dataframe_to_csv
from tqdm    import tqdm

import matplotlib.pyplot as plt

import gc; gc.enable()
import warnings; warnings.filterwarnings("ignore")

class Stats(object):


    def reset_trade_files(self, train_fnm, test_fnm):
        self.train_fnm = train_fnm
        self.train_df = self.__read_trades__(train_fnm)
        assert len(self.train_df) > 0, "training data set cannot be empty"
        gc.collect()

        self.test_fnm  = test_fnm
        self.test_df = self.__read_trades__(test_fnm)
        gc.collect()

    def __init__(self, train_fnm, test_fnm):
        self.reset_trade_files(train_fnm, test_fnm)
        self.ticker = ''
        self.batched_days = []

    def __read_trades__(self, fnm):
        possible_trades_df = pd.read_csv(fnm)
        return possible_trades_df

    def __daily_return__(self, pct_gain, days):
        daily_return = (1+pct_gain/100) ** (1/(days)) - 1
        daily_return = round(daily_return * 100.0, 2)
        return daily_return

    def __daily_return_from_df__(self, ticker_df, pct_col):
        ticker_df['ret'] = 1 + ticker_df[pct_col]/100
        net_gain         = ticker_df.ret.product() - 1
        net_gain         = net_gain * 100

        if net_gain <= -100:
            return -np.Inf

        days = int(ticker_df.trading_days.sum())
        return self.__daily_return__(net_gain, days)

    def __init_gain_ticker_stats__(self):
        self.cnt_gain      = 0
        self.min_pct_gain  = 0
        self.max_pct_gain  = 0
        self.std_pct_gain  = 0
        self.mean_pct_gain = 0
        self.day_gain      = 0
        self.gain_daily_return = 0

    def __init_loss_ticker_stats__(self):
        self.cnt_loss      = 0
        self.min_pct_loss  = 0
        self.max_pct_loss  = 0
        self.std_pct_loss  = 0
        self.mean_pct_loss = 0
        self.day_loss      = 0
        self.loss_daily_return = 0

    def __init_zero_ticker_stats__(self):
        self.cnt_zero      = 0
        self.mean_day_zero = 0

    def __init_tot_ticker_stats__(self):
        self.total_days = 0
        self.total_cnt  = 0
        self.daily_ret  = -np.Inf

    def __init_ticker_stats__(self):
        self.__init_gain_ticker_stats__()
        self.__init_loss_ticker_stats__()
        self.__init_zero_ticker_stats__()
        self.__init_tot_ticker_stats__()

    def __calc_gain_ticker_stats__(self):
        idx = (self.train_df.ticker == self.ticker) \
            & (self.train_df.gain > 0)
        ticker_df = self.train_df.loc[idx]
        if len(ticker_df) == 0:  
            return

        pct_col = 'pct_gain'
        ticker_df[pct_col] = (ticker_df.gain / ticker_df.buy_close) * 100

        self.cnt_gain       = len(ticker_df)
        self.min_pct_gain   = ticker_df.pct_gain.min()
        self.max_pct_gain   = ticker_df.pct_gain.max()
        self.std_pct_gain   = ticker_df.pct_gain.std()
        self.mean_pct_gain  = ticker_df.pct_gain.mean()
        self.mean_day_gain  = float(ticker_df.trading_days.mean())
        self.gain_daily_ret = self.__daily_return_from_df__(ticker_df, pct_col)

    def __calc_loss_ticker_stats__(self):
        idx = (self.train_df.ticker == self.ticker) \
            & (self.train_df.gain < 0)
        ticker_df = self.train_df.loc[idx]
        if len(ticker_df) == 0:  
            return

        pct_col = 'pct_loss'
        ticker_df[pct_col] = (ticker_df.gain / ticker_df.buy_close) * 100

        self.cnt_loss       = len(ticker_df)
        self.min_pct_loss   = ticker_df.pct_loss.min()
        self.max_pct_loss   = ticker_df.pct_loss.max()
        self.std_pct_loss   = ticker_df.pct_loss.std()
        self.mean_pct_loss  = ticker_df.pct_loss.mean()
        self.mean_day_loss  = float(ticker_df.trading_days.mean())
        self.loss_daily_ret = self.__daily_return_from_df__(ticker_df, pct_col)

    def __calc_zero_ticker_stats__(self):
        idx = (self.train_df.ticker == self.ticker) \
            & (self.train_df.gain == 0)
        ticker_df = self.train_df.loc[idx]

        self.cnt_zero = len(ticker_df)
        if self.cnt_zero > 0:
            self.mean_day_zero = float(ticker_df.trading_days.mean())

    def __calc_tot_ticker_stats__(self): 
        idx = (self.train_df.ticker == self.ticker)
        ticker_df = self.train_df.loc[idx]
        if len(ticker_df) == 0:
            return

        pct_col = 'pct_gain'
        ticker_df[pct_col] = (ticker_df.gain / ticker_df.buy_close) * 100

        self.total_cnt  = self.cnt_gain + self.cnt_loss + self.cnt_zero
        self.total_days = int(ticker_df.trading_days.sum())
        self.daily_ret = self.__daily_return_from_df__(ticker_df, pct_col)

        desired_cnt   = len(ticker_df[ticker_df.daily_return > 0.5])
        len_ticker_df = len(ticker_df)
        pct_desired   = (desired_cnt / len_ticker_df) * 100
        self.pct_desired   = round(pct_desired, 2)

        self.gain_ratio = self.cnt_gain / self.total_cnt       

    def __ticker_stats_to_df__(self):
        t_dict = {
            'ticker'        : [self.ticker], 

            'cnt_gain'      : [self.cnt_gain],
            'min_pct_gain'  : [self.min_pct_gain],
            'max_pct_gain'  : [self.max_pct_gain],
            'std_pct_gain'  : [self.std_pct_gain], 
            'mean_pct_gain' : [self.mean_pct_gain],
            'mean_day_gain' : [self.mean_day_gain],   
            'gain_daily_ret': [self.gain_daily_ret],

            'cnt_loss'      : [self.cnt_loss],
            'min_pct_loss'  : [self.min_pct_loss],
            'max_pct_loss'  : [self.max_pct_loss],
            'std_pct_loss'  : [self.std_pct_loss], 
            'mean_pct_loss' : [self.mean_pct_loss],
            'mean_day_loss' : [self.mean_day_loss],   
            'loss_daily_ret': [self.loss_daily_ret],
            
            'cnt_zero'      : [self.cnt_zero],
            'mean_day_zero' : [self.mean_day_zero],   
 
            'total_cnt'     : [self.cnt_gain + self.cnt_loss + self.cnt_zero],
            'total_days'    : [self.total_days],
            'mean_day'      : [self.total_days / self.total_cnt],
            'daily_ret'     : [self.daily_ret],

            'pct_desired'   : [self.pct_desired],
            'gain_ratio'    : [self.gain_ratio],
            'good'          : [ 0 ]
            }

        return dict_to_dataframe(t_dict, STAT_COL_TYPES)

    def __calc_ticker_stats__(self):
        self.__init_ticker_stats__()

        self.__calc_gain_ticker_stats__()
        self.__calc_loss_ticker_stats__()
        self.__calc_zero_ticker_stats__()
        self.__calc_tot_ticker_stats__()
        gc.collect()
        
        return self.__ticker_stats_to_df__()
    
    def calc_stats(self):
        self.train_df['gain'] = self.train_df.sell_close - self.train_df.buy_close
        tickers = list(self.train_df.ticker.unique())

        self.df = empty_dataframe(STAT_COLS, STAT_COL_TYPES)
        save_ticker = self.ticker
        for t in tqdm(tickers, "Stats:"):
            gc.collect()
            self.ticker = t
            self.df = pd.concat( [self.df, self.__calc_ticker_stats__()], 
                                sort=False )

        self.ticker = save_ticker
        log('')
        log(f'\n{self.df.describe()}')

    def plot_trades(self):
        fig, axs = plt.subplots(2)

        self.train_df.plot.scatter(x='trading_days', y='gain_pct', ax=axs[0], 
                                   figsize=(18,12))

        self.test_df.plot.scatter(x='trading_days',  y='gain_pct', ax=axs[1], 
                                  figsize=(18,12))

        title_str = 'Possible trades: trading days vs percentage gain'
        axs[0].set_title(title_str + '(training set)')
        axs[1].set_title(title_str + '(test set)')

        plt.savefig(f'{PICPATH}stats_scatter.png')

    def __sum_n_count__(self, df):
        if len(df) == 0:
            return 0, 0

        aggs = ['sum', 'count']
        s, c = df['gain'].agg(aggs)
        return s, c

    def __calc_n_print_ratios__(self, df):
        gain_s, gain_c = self.__sum_n_count__(df.loc[df.gain > 0])
        loss_s, loss_c = self.__sum_n_count__(df.loc[df.gain < 0])
        zero_s, zero_c = self.__sum_n_count__(df.loc[df.gain == 0])

        gain_s = round(gain_s, 2)
        loss_s = round(loss_s, 2)

        log('')
        log(f'gain_sum={gain_s} gain_cnt={gain_c}')
        log(f'loss_sum={loss_s} loss_cnt={loss_c}')
        log(f'zero_sum={zero_s} zero_cnt={zero_c}')
        log('')
        gain_share = (gain_c / (gain_c + loss_c + zero_c)) * 100
        gain_share = round(gain_share,2)
        log(f'gain_share={gain_share} %')
        log('')

    def train_df_ratios(self):
        log('ratios for all tickers self.train_df:')
        self.__calc_n_print_ratios__(self.train_df)

    def __good_tickers_to_index__(self):
        self.idx = self.df.ticker == ''
        for t in self.good_tickers:
            self.idx |= self.df.ticker == t

    def __log_heuristics_stats__(self):
        log(f'len(self.df)={len(self.df)}')
        log(f'len(self.df[idx])={len(self.df[self.idx])}')
        ratio = round(len(self.df[self.idx])/len(self.df), 2)
        log(f'ratio={ratio}')

    def heuristic(self, threshold, verbose=True):
        self.threshold = threshold
        idx = (self.df.daily_ret  > 0.0) \
            & (self.df.pct_desired > threshold)
        self.good_tickers = list(self.df.ticker.loc[idx].unique())
        log(f'len(self.good_tickers)={len(self.good_tickers)}')

        self.__good_tickers_to_index__()
        if verbose == True:
            self.__log_heuristics_stats__()


    def __good_subset_posible_trades__(self):
        cols = list(self.train_df.columns)
        self.good_df = empty_dataframe(TRADE_COLS, TRADE_COL_TYPES)
        for t in self.good_tickers:
            gc.collect()
            idx = self.train_df.ticker == t
            df = self.train_df.loc[idx]
            if len(df) > 0:
                self.good_df = pd.concat([self.good_df, df])

    def good_ticker_ratios(self):
        self.__good_subset_posible_trades__()
        log('ratios for good tickers:')
        self.__calc_n_print_ratios__(self.good_df)

    def __log_save_stats__(self):

        good_len = len(self.good_df)
        train_df_len = len(self.train_df)

        df_len = len(self.df)
        df_good_sum = sum(self.df.good)

        share= round(good_len / train_df_len * 100, 0)

        log('')
        log(f'total ticker count={df_len}')
        log(f'good tickers count={df_good_sum}')
        log('')
        log(f'all ticker trades ={train_df_len}')
        log(f'good ticker trades={good_len}')
        log(f'good ticker share ={share}%')
        log('')

    def __set_good_tickers__(self):
        self.df['good'] = 0
        for t in self.good_tickers:
            self.df.good.loc[self.df.ticker == t] = 1

    def save_good_tickers(self, fnm):

        self.__set_good_tickers__()
        self.__log_save_stats__()
        save_dataframe_to_csv(self.df, fnm)

    def add_day(self, trading_date):

        idx = self.test_df.sell_date == trading_date
        test_len = len(self.test_df.loc[idx])
        if test_len == 0:
            return
        
        self.batched_days.append(trading_date)
        if len(self.batched_days) < STAT_BATCH_SIZE:
            return

        idx = self.test_df.sell_date == ''
        for d in self.batched_days:
            idx |= self.test_df.sell_date == d

        tickers_to_process = list(self.test_df.loc[idx].ticker.unique())
        log(f'Processing {self.batched_days}')
        log(f'Unique tickers={len(tickers_to_process)}')

        self.train_df = pd.concat([self.train_df, self.test_df.loc[idx]])

        save_ticker = self.ticker
        for t in tickers_to_process:
            self.ticker = t
            idx = self.df.ticker == t
            self.df = self.df[~idx]
            ticker_df = self.__calc_ticker_stats__()
            self.df = pd.concat([self.df, ticker_df])

        self.ticker = save_ticker
        self.heuristic(self.threshold)
        self.__set_good_tickers__()
        self.batched_days = []

def test_add_day(stats):
    log('Test add_day() functionality for non-existing trading date:')
    stats.add_day('2100-04-28')

    dates_to_process = sorted(list(stats.test_df.sell_date.unique()))

    for d in tqdm(dates_to_process, "Dates:"):
        stats.add_day(d)

def stats_main():
    log("Starting stats.py")
    log('')
    stats = Stats(TRAIN_TRADE_FNM, TEST_TRADE_FNM)

    log('Calculating individual ticker stats...\n')
    stats.calc_stats()

    log('Plotting the self.train_df and self.test_df scatter plots...\n')
    stats.plot_trades()

    log('Printing the stats.df ratios...\n')
    stats.train_df_ratios()

    log('Determining the good tickers and printing the ratios...\n')
    stats.heuristic(10)
    stats.good_ticker_ratios()

    log('Saving the ticker stats (including good)...')
    stats.save_good_tickers(STATS_FNM)

    test_add_day(stats)

if __name__ == "__main__":

    log_fnm = "stats"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        stats_main()
