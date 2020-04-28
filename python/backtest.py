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

import pandas                as pd
import numpy                 as np
import matplotlib.pyplot     as plt
import gc; gc.enable()

from tqdm                    import tqdm
from time                    import sleep
from datetime                import datetime, timedelta

from util                    import open_logfile, log, get_stock_period, \
                                    build_ticker_list, is_holiday, \
                                    get_current_day_and_time, add_days_to_date, \
                                    set_to_string, add_spaces

from pnl                     import Capital, PnL
from symbols                 import TOLERANCE, DATAPATH, LOGPATH, PICPATH, \
                                    STOP_LOSS, STATS_FNM, TEST_TRADE_FNM, \
                                    BUY_FNM, TRADE_COLS

import warnings; warnings.filterwarnings("ignore")

class Backtester(object):

    def augment_possible_trades_with_buy_opportunities(self):
        cols =[ 'ticker', 'mean_pct_gain', 'mean_day', self.ret_col ]
        idx = self.buy_opportunities_df.buy_date >= \
              max(self.possible_trades_df.buy_date)
        br_df = pd.merge(self.buy_opportunities_df[idx], 
                         self.ticker_stats_df[cols], how='inner')

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

        self.possible_trades_df = pd.concat([self.possible_trades_df, br_df])
        gc.collect()

    def __init__(self):
        log(f"Initializing backtester class: ")

        # Read ticker statistics
        self.ticker_stats_df = pd.read_csv(STATS_FNM)
        idx = self.ticker_stats_df.good == 1
        self.tickers = self.ticker_stats_df[idx].ticker.to_list()
        log(f"tickers={self.tickers} len(tickers)={len(self.tickers)}\n\n")

        # Read possible trades and buy opportunities
        self.possible_trades_df = pd.read_csv(TEST_TRADE_FNM)
        self.buy_opportunities_df = pd.read_csv(BUY_FNM)

        # Augment buy recommendations
        self.ret_col = 'daily_ret' #'gain_daily_ret'
        self.augment_possible_trades_with_buy_opportunities()

        # Determine start and end date for backtesting period.
        self.start_date = min(self.possible_trades_df.buy_date)
        self.end_date   = max(self.possible_trades_df.sell_date)
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
    
    def pct_desired(self, threshold):
        # Read ticker statistics
        self.threshold = threshold
        self.ticker_stats_df = pd.read_csv(STATS_FNM)
        idx = (self.ticker_stats_df.pct_desired >= threshold) \
            & (self.ticker_stats_df.daily_ret > 0)
        self.tickers = self.ticker_stats_df[idx].ticker.to_list()
        self.len_tickers = len(self.tickers)
        log(f"pct_desired({threshold}):len_tickers={self.len_tickers}\n\n")

    def log_invested(self, message):
        log(message)
        log(f"invested={list(self.myPnL.invested.keys())}"
            f" ({len(self.myPnL.invested)})")

    def log_PnL(self, message):
        log(message)
        log(f"capital={self.myPnL.capital} in_use={self.myPnL.in_use}"
            f" free={self.myPnL.free}")
    
    # TODO: rewrite when implementing dynamic ticker_stats_df updates
    def sort_possible_trades(self):
        self.possible_trades_df = pd.merge(self.possible_trades_df, 
            self.ticker_stats_df[['ticker', self.ret_col]], how='inner')
        self.possible_trades = self.possible_trades_df.sort_values(
            by=['buy_date', self.ret_col], ascending=[True, False])
        self.possible_trades = self.possible_trades.reset_index()

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
        self.sort_possible_trades()

        self.sell_dates   = {}

        log('', True)
        log(f"Possible trades to simulate: {len(self.possible_trades)}", True)
        log(f"Trading days to simulate   : "
            f"{len(self.backtest_trading_dates)}", True)
        log(f'Pct_desired threshold      : {self.threshold}\n', True)

    def extract_dates_n_ticker(self):
        idx = self.i_possible_trades
        self.buy_date  = self.possible_trades.buy_date.iloc[idx]
        self.sell_date = self.possible_trades.sell_date.iloc[idx]
        self.ticker    = self.possible_trades.ticker.iloc[idx]

    def next_possible_trade_index(self):
        while self.i_possible_trades < len(self.possible_trades)-1:
            self.i_possible_trades = self.i_possible_trades + 1
            idx = self.i_possible_trades
            t = self.possible_trades.ticker.iloc[idx]
            if t in self.tickers:
                return True
        return False

    def calc_expected_gain(self):
        idx = self.ticker_stats_df.ticker == self.ticker
        total_days = float(self.ticker_stats_df.total_days.loc[idx])
        total_cnt  = float(self.ticker_stats_df.total_cnt.loc[idx])
        expected_days = total_days / total_cnt
        daily_ret = float(self.ticker_stats_df[self.ret_col].loc[idx])
        expected_gain = (1 + (daily_ret/100)) ** expected_days - 1 
        return expected_gain

    def find_lowest_expected_gain(self):
        lowest_expected_gain = None 
        lowest_ticker        = None

        save_ticker = self.ticker
        for t in self.myPnL.invested.keys():
            self.ticker = t
            expected_gain = self.calc_expected_gain()

            if lowest_expected_gain is None or \
               expected_gain < lowest_expected_gain:
                lowest_expected_gain = expected_gain
                lowest_ticker        = t

        self.ticker = save_ticker
        return lowest_ticker, lowest_expected_gain

    def find_stock_to_replace(self):
        expected_gain = self.calc_expected_gain()
        lowest_ticker, lowest_expected_gain = self.find_lowest_expected_gain()

        if lowest_expected_gain is not None and \
           expected_gain > lowest_expected_gain:
            self.myPnL.sell_stock(lowest_ticker, self.trading_date)

    def calc_amount(self):
        self.amount = self.myPnL.capital / self.max_stocks

    def buy_stock(self):
        self.myPnL.buy_stock(self.ticker, self.buy_date, 
                self.sell_date, self.amount)

        if self.sell_date in self.sell_dates:
            self.sell_dates[self.sell_date].append(self.ticker)
        else:
            self.sell_dates[self.sell_date] = [ self.ticker ]

    def buy_stock_if_possible(self):
        # ignore buy opportunity if already invested in the stock
        # and don't buy the stock
        if self.ticker in self.myPnL.invested:
            return

        # Only attempt to buy a stock when below max # stocks and 
        # we have enough free money to buy at least 25% of stock 
        if len(self.myPnL.invested) < self.max_stocks and \
            self.myPnL.free >= self.amount*0.25:
            self.buy_stock()

    def sell_stocks(self, to_sell):
        for ticker in to_sell:
            if ticker in self.myPnL.invested:
                self.myPnL.sell_stock(ticker, self.trading_date)

    def process_sell_signals(self):
        if self.trading_date in self.sell_dates:
            to_sell = self.sell_dates.pop(self.trading_date, [])
            self.sell_stocks(to_sell)

    def process_buy_opportunities(self):
        if self.i_possible_trades >= len(self.possible_trades):
            return
        
        self.extract_dates_n_ticker()
        while self.trading_date == self.buy_date:
            idx = self.ticker_stats_df.ticker == self.ticker
            self.daily_ret = float(self.ticker_stats_df[self.ret_col].loc[idx])
            if self.daily_ret > 0:
                self.calc_amount()
                if len(self.myPnL.invested) >= self.max_stocks:
                    self.find_stock_to_replace()

                self.buy_stock_if_possible()

            # move to next possible trading opportunity
            if self.next_possible_trade_index() == False:
                break

            self.extract_dates_n_ticker()

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


    def update_stats_for_ticker(self, ticker_df):
        ticker = ticker_df.ticker.unique()
        assert len(ticker_df) == 1, "expected to be one row..."
        tdf = ticker_df.copy()
        tdf['gain'] = tdf.sell_close - tdf.buy_close
        gain = tdf.gain.max()

        # to be expanded ...

    def update_ticker_stats(self):
        idx = self.possible_trades.sell_date == self.trading_date
        df = self.possible_trades.loc[idx].copy()
        if len(df) == 0:
            return

        tickers_to_check = list(df.ticker.unique())
        for t in tickers_to_check:
            if t in self.tickers:
                idx = df.ticker == t
                self.update_stats_for_ticker(df.loc[idx])

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
            self.update_ticker_stats()

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
        log(f'Saving scatter plot self.possible_trades_df to {fnm}')
        self.scatter_plot(self.possible_trades_df, 'trading_days', fnm)

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
    bt = Backtester()

    capital_dict = {}
    invested_dict = {}
    len_tickers_dict = {}
    gains_dict = {}
    loss_dict = {}

    # thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    thresholds = [10, 20, 30]
    for th in thresholds:
        bt.pct_desired(th)
        bt.run_back_test(10000, 5)
        log('', True)
        bt.log_PnL('myPnL=')
        bt.log_invested('invested in:')
        gains, losses = bt.print_backtest_stats()

        # save key stats of run 
        capital_dict[th]      = bt.myPnL.capital
        invested_dict[th]     = bt.myPnL.invested.keys()
        len_tickers_dict[th]  = bt.len_tickers
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

if __name__ == "__main__":

    log_fnm = "test"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        backtest_main()