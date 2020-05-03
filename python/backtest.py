#
# Stockie - Stock Trading System
#
# Initial version of backtesting module. It allows you to backtest as well as 
# to get buy and sell recommendations (still to be implemented as part of final
# project). The backtest_xxx.ipynb notebooks depend on the functionality 
# provided
# by this module.
#
# Author:   Frank Kornet
# Date:     16 April 2020
# Version:  0.2
#

import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
import gc; gc.enable()

from tqdm                   import tqdm
from time                   import sleep, time
from datetime               import datetime, timedelta

from util                   import open_logfile, log, get_stock_period, \
                                   build_ticker_list, is_holiday, \
                                   get_current_day_and_time, add_days_to_date, \
                                   set_to_string, add_spaces, dict_to_dataframe

from pnl                    import Capital, PnL
from symbols                import TOLERANCE, DATAPATH, LOGPATH, PICPATH, \
                                   STOP_LOSS, STATS_FNM, TEST_TRADE_FNM, \
                                   TRAIN_TRADE_FNM, BUY_FNM, TRADE_COLS, \
                                   BT_DRET_COL
from stats                  import Stats

import warnings; warnings.filterwarnings("ignore")

class Backtester(object):

    def augment_possible_trades_with_buy_opportunities(self):
        cols =[ 'ticker', 'mean_pct_gain', 'mean_day', self.ret_col ]
        idx = self.buy_opportunities_df.buy_date >= \
              max(self.stats.test_df.buy_date)
        br_df = pd.merge(self.buy_opportunities_df[idx], 
                         # self.ticker_stats_df[cols], how='inner')
                         self.stats.df[cols], how='inner')

        br_df['gain_pct']= br_df.mean_pct_gain.astype(float)
        br_df['trading_days'] = round(br_df.mean_day, 0).astype(int)
        br_df['daily_return'] = br_df[self.ret_col].astype(float)

        br_df['sell_date'] = br_df[['buy_date', 'trading_days']].apply(
             lambda x: pd.to_datetime(x.buy_date)+timedelta(x.trading_days), 
             axis=1)
        br_df.sell_date = br_df.sell_date.astype(str)


        br_df['sell_close'] = round(br_df.buy_close*(1+br_df.mean_pct_gain/100),2)
        br_df = br_df[TRADE_COLS]

        log('')
        cols = ['buy_date', 'daily_return']
        log('Adding following buy opportunities to possible trades:\n'
            f'{br_df.sort_values(by=cols, ascending=[True, False])}')
        log('')

        self.stats.test_df = pd.concat([self.stats.test_df, br_df])
        gc.collect()

    def __init__(self, keep="3m", threshold=20, update_stats=True):
        log(f"Initializing backtester class: ")

        # Read ticker statistics
        # self.ticker_stats_df = pd.read_csv(STATS_FNM)
        # idx = self.ticker_stats_df.good == 1
        # self.tickers = self.ticker_stats_df[idx].ticker.to_list()
        # log(f"tickers={self.tickers} len(tickers)={len(self.tickers)}\n\n")

        # Create a Stats instance and create ticker stats for it
        self.stats = Stats(TRAIN_TRADE_FNM, TEST_TRADE_FNM)
        self.stats.calc_stats()
        self.threshold = threshold
        self.keep = keep
        self.update_stats = update_stats
        self.stats.heuristic(self.threshold, verbose=False)

        # Read buy opportunities
        self.buy_opportunities_df = pd.read_csv(BUY_FNM)

        # Augment buy recommendations
        self.ret_col = BT_DRET_COL
        self.augment_possible_trades_with_buy_opportunities()

        # Determine start and end date for backtesting period.
        self.start_date = min(self.stats.test_df.buy_date)
        self.end_date   = max(self.stats.test_df.sell_date)
        self.start_date = add_days_to_date(self.start_date, -5)
        self.end_date   = add_days_to_date(self.end_date, 5)

        # Pull down MSFT stock for period and use that as basis for determining
        # the stock market trading days
        success, hist = get_stock_period('MSFT', 2, "max")
        assert success == True, 'Failed to retrieve MSFT data'
        idx = (hist.index >= self.start_date) & (hist.index <= self.end_date)
        self.backtest_trading_dates = hist.loc[idx].index.to_list()
        self.start_date = min(self.backtest_trading_dates)
        self.end_date   = max(self.backtest_trading_dates)
    
    # TODO: logic for filtering is really in buy_stock
    def pct_desired(self, threshold):
        # Read ticker statistics
        self.threshold = threshold
        self.stats.heuristic(self.threshold, verbose=False)
        # self.ticker_stats_df = pd.read_csv(STATS_FNM)
        # idx = (self.ticker_stats_df.pct_desired >= threshold) \
        #     & (self.ticker_stats_df.daily_ret > 0)
        # self.tickers = self.ticker_stats_df[idx].ticker.to_list()
        # self.len_tickers = len(self.tickers)
        # log(f"pct_desired({threshold}):len_tickers={self.len_tickers}\n\n")

    def log_invested(self, message):
        log(message)
        log(f"invested={list(self.myPnL.invested.keys())}"
            f" ({len(self.myPnL.invested)})")

    def log_PnL(self, message):
        log(message)
        log(f"capital={self.myPnL.capital} in_use={self.myPnL.in_use}"
            f" free={self.myPnL.free}")


    def init_back_test_run(self, capital, max_stocks):
        log("starting backtester")

        # Initialize the key variable
        self.capital    = capital
        self.free       = capital
        self.in_use     = 0
        self.max_stocks =  max_stocks
        self.myPnL   = PnL( self.start_date, self.end_date, 
                            self.capital, self.in_use, self.free, 
                            self.max_stocks)

        # Sort the possible trades so they are processed in order
        self.i_possible_trades = 0
        #self.sort_possible_trades()

        self.sell_dates   = {}

        log('', True)
        log(f"Possible trades to simulate: {len(self.stats.test_df)}", True)
        log(f"Trading days to simulate   : "
            f"{len(self.backtest_trading_dates)}", True)
        log(f'Pct_desired threshold      : {self.threshold}\n', True)
    
    def calc_amount(self):
        self.amount = self.myPnL.capital / self.max_stocks

    def buy_stock(self):
        self.myPnL.buy_stock(self.ticker, self.buy_date, 
                self.sell_date, self.amount)

        # something went wrong (e.g. unable to retrieve hist data)
        if self.ticker not in self.myPnL.invested.keys():
            return
        
        if self.sell_date in self.sell_dates:
            self.sell_dates[self.sell_date].append(self.ticker)
        else:
            self.sell_dates[self.sell_date] = [ self.ticker ]


    def sell_stocks(self, to_sell):
        for ticker in to_sell:
            if ticker in self.myPnL.invested:
                self.myPnL.sell_stock(ticker, self.trading_date)

    def process_sell_signals(self):
        if self.trading_date in self.sell_dates:
            to_sell = self.sell_dates.pop(self.trading_date, [])
            self.sell_stocks(to_sell)

    def buy_stocks(self, to_buy):
        self.buy_date = self.trading_date
        for t in to_buy:
            self.ticker = t
            self.calc_amount()
            idx = (self.stats.test_df.ticker == t) \
                & (self.stats.test_df.buy_date == self.buy_date)
            self.sell_date = self.stats.test_df.sell_date.loc[idx].min()
            self.buy_stock()

    def process_buy_opportunities(self):

        own_set = set(self.myPnL.invested.keys())
        o_dict = {'ticker' : list(own_set), 'own' : [1]*len(own_set)}
        odf = dict_to_dataframe(o_dict, [str, int])

        cols = ['ticker', self.ret_col]
        odf = pd.merge(left=odf, 
                        # right=self.ticker_stats_df[cols],
                        right=self.stats.df[cols],
                        on='ticker', how='inner')

        # create dataframe for possible buys        
        cols = ['ticker', 'sell_date']
        self.buy_date = self.trading_date
        idx = self.stats.test_df.buy_date == self.buy_date
        tdf = self.stats.test_df[cols].loc[idx].copy()
        tdf['own'] = 0

        cols = ['ticker', self.ret_col, 'pct_desired']
        tdf = pd.merge(left=tdf, 
                        # right=self.ticker_stats_df[cols],
                        right=self.stats.df[cols],
                        on='ticker', how='inner')

        # only consider 'good' ticker for buying
        idx = (tdf[self.ret_col] > 0.0) \
            & (tdf.pct_desired >= self.threshold)
        tdf=tdf.loc[idx]

        # concat, sort and determine final_set
        df = pd.concat([odf, tdf]).sort_values(by=self.ret_col, ascending=False)
        #log(df, True)
        final_set = set(df.ticker.iloc[0:self.max_stocks])

        to_sell = own_set - final_set
        self.sell_stocks(to_sell)

        to_buy = final_set - own_set
        self.buy_stocks(to_buy)

    def close_day(self):
        day = self.trading_date
        self.myPnL.day_close(day)

        start_invested = set(self.start_invested.keys())
        end_invested   = set(self.end_invested.keys())

        hold = set_to_string(start_invested & end_invested, 30)
        sell = set_to_string(start_invested - end_invested, 30)
        buy  = set_to_string(end_invested - start_invested, 30)

        cap  = int(round(self.myPnL.capital/1000, 0))
        free = int(round(self.myPnL.free/1000, 0))
        use  = int(round(self.myPnL.in_use/1000, 0))

        log(f'{day}\t{cap}\t{free}\t{use}\t{hold}\t{sell}\t{buy}')

    def print_heading(self):
        width = 30
        h = add_spaces("hold", width)
        s = add_spaces("sell", width)
        b = add_spaces("buy", width)
        u = "=" * width

        log(f'day\t\tcapital\tfree\tin_use\t{h}\t{s}\t{b}')
        log(f'===\t\t=======\t====\t======\t{u}\t{u}\t{u}')

    def main_back_test_loop(self):

        self.print_heading()

        for self.trading_day, self.trading_date in \
            enumerate(tqdm(self.backtest_trading_dates, 
                           desc="simulate trades: ")):
            
            gc.collect()
            self.trading_date = str(self.trading_date)[:10]
            self.start_invested = self.myPnL.invested.copy()
            self.process_sell_signals()
            self.process_buy_opportunities()
            self.end_invested = self.myPnL.invested.copy()
            self.close_day()
            if self.update_stats == True:
                self.stats.add_day(self.trading_date)

    def run_back_test(self, capital, max_stocks):
        self.init_back_test_run(capital, max_stocks)
        self.main_back_test_loop()
        self.myPnL.df.days_in_trade = self.myPnL.df.days_in_trade.astype(int)

    def create_sell_df(self):
        gc.collect()
        idx                  = self.myPnL.df.action=='SELL'
        sell_df              = self.myPnL.df[idx].copy()
        sell_df['gain']      = (sell_df.close_amount - sell_df.orig_amount)
        sell_df['gain_pct']  = round((sell_df.gain/sell_df.orig_amount)*100, 2)
        sell_df['daily_ret'] = ( (  (1 + sell_df.gain_pct/100) \
                                ** (1/ sell_df.days_in_trade) ) - 1 ) * 100

        self.sell_df = sell_df

    def describe_sell_df(self):
        log('')
        log(f'len(self.sell_df)={len(self.sell_df)}')
        log('')
        cols = ['gain', 'gain_pct', 'daily_ret']
        log(f'self.sell_df.describe():\n{self.sell_df[cols].describe()}')
        log('')

    def value_counts_sell_df(self):
        log('')
        log('Value counts sell_df,days_in_trade:')
        log(f'\n{self.sell_df.days_in_trade.value_counts()}')
        log('')

    def scatter_plot(self, df, day_col, fnm):
        if len(df) > 0:
            df.plot.scatter(x=day_col, y='gain_pct', figsize=(18,8))
            plt.savefig(fnm)
        else:
            log(f'Data frame is empty: unable to scatter plot!')

    def sell_df_scatter_plot(self):
        log('')
        fnm = f'{PICPATH}sell_df_scatter_{self.threshold}.png'
        log(f'Saving scatter plot self.sell_df to {fnm}')
        self.scatter_plot(self.sell_df, 'days_in_trade', fnm)

    def possible_trades_df_scatter_plot(self):
        log('')
        fnm = f'{PICPATH}pos_trades_df_scatter_{self.threshold}.png'
        log(f'Saving scatter plot self.stats.test_df to {fnm}')
        self.scatter_plot(self.stats.test_df, 'trading_days', fnm)

    def calc_sum_n_count(self, df):
        s, c =  df['gain'].agg(['sum', 'count'])
        return int(round(s,0)), int(c)

    def calc_actual_gains_losses_zeros(self):
        df = self.sell_df
        gain_sum, gain_cnt = self.calc_sum_n_count(df[df.gain >  0])
        loss_sum, loss_cnt = self.calc_sum_n_count(df[df.gain <  0])
        zero_sum, zero_cnt = self.calc_sum_n_count(df[df.gain == 0])

        log('')
        log(f'gain_sum={gain_sum}\t\tgain_cnt={gain_cnt}')
        log(f'loss_sum={loss_sum}\t\tloss_cnt={loss_cnt}')
        log(f'zero_sum={zero_sum}\t\tzero_cnt={zero_cnt}')
        log('')
        log(f'Total # transactions : {gain_cnt+loss_cnt+zero_cnt}')

        return gain_sum, loss_sum

    def print_backtest_stats(self):
        self.create_sell_df()
        self.describe_sell_df()
        self.value_counts_sell_df()
        self.sell_df_scatter_plot()
        self.possible_trades_df_scatter_plot()
        gains, losses = self.calc_actual_gains_losses_zeros()

        fnm = f'{DATAPATH}actual_{self.threshold}.csv'
        log(f'Saving sell_df to {fnm}')
        self.sell_df.to_csv(fnm, index=False)
        return gains, losses

def backtest_main():
    start_time = time()

    bt = Backtester(update_stats=False)

    capital_dict = {}
    invested_dict = {}
    len_tickers_dict = {}
    gains_dict = {}
    loss_dict = {}

    # thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    thresholds = [20]
    for th in thresholds:
        if th >= 10:
            bt.stats.reset_trade_files(TRAIN_TRADE_FNM, TEST_TRADE_FNM)            
        bt.pct_desired(th)
        bt.run_back_test(10000, 5)
        log('', True)
        bt.log_PnL('myPnL=')
        bt.log_invested('invested in:')
        gains, losses = bt.print_backtest_stats()

        # save key stats of run 
        capital_dict[th]      = bt.myPnL.capital
        invested_dict[th]     = bt.myPnL.invested.keys()
        good_tickers = bt.stats.df.loc[bt.stats.df.good == 1].ticker
        len_tickers_dict[th]  = len(good_tickers)
        gains_dict[th]        = gains
        loss_dict[th]         = losses

        bt.myPnL.myCapital.df.index = bt.myPnL.myCapital.df.date
        to_plot_cols = ['capital', 'in_use']
        bt.myPnL.myCapital.df[to_plot_cols].plot(figsize=(18,10))
        plt.savefig(f'{PICPATH}trade_threshold_{th}.png')

    # Print summary...
    log('', True)
    log('Threshold\tCapital\tGains\tLosses\tReturn\tLen\tInvested', True)
    log('=========\t=======\t=====\t======\t======\t===\t========', True)
    trading_days = len(bt.backtest_trading_dates)

    for th in thresholds:
        cap   = capital_dict[th]
        ret   = ( (cap/10000) ** (1/trading_days)) - 1
        ret   = round (ret * 100, 2)
        inv   = set_to_string(list(invested_dict[th]), 30)
        length   = len_tickers_dict[th]
        cap   = int(round(cap/1000,0))
        gains = int(round(gains_dict[th]/1000,0))
        losses = int(round(loss_dict[th]/1000,0))
        log(f'{th}\t\t{cap}\t{gains}\t{losses}\t{ret}%\t{length}\t{inv}', True)

    log('')
    log('Done.')

    seconds = int(time() - start_time)
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    hours = int(minutes / 60)
    minutes = minutes - hours * 60
    log(f'Run time (hh:mm:ss) : {hours:02d}:{minutes:02d}:{seconds:02d}', True)

if __name__ == "__main__":

    log_fnm = "test"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        backtest_main()