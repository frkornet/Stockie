#
# Stockie - Stock Trading System
#
# Initial version of backtesting module. It allows you to backtest as well as 
# to get buy and sell recommendations (still to be implemented as part of final
# project). The backtest_xxx.ipynb notebooks depend on the functionality provided
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

from util                    import open_logfile, log, get_stock, build_ticker_list, \
                                    get_current_day_and_time, is_holiday, add_days_to_date
from pnl                     import Capital, PnL
from symbols                 import TOLERANCE, DATAPATH, LOGPATH, STOP_LOSS, \
                                    STATS_FNM, TRADE_FNM, BUY_FNM

import warnings; warnings.filterwarnings("ignore")

class Backtester(object):

    def augment_possible_trades_with_buy_opportunities(self):
        cols =[ 'ticker', 'pct_gain', 'day_gain', self.ret_col ]
        br_df = pd.merge(self.buy_opportunities_df, 
                         self.ticker_stats_df[cols], how='inner')

        br_df['gain_pct']     = br_df.pct_gain.astype(float)
        br_df['trading_days'] = round(br_df.day_gain, 0).astype(int)
        br_df['daily_return'] = br_df[self.ret_col].astype(float)

        br_df['sell_date'] = br_df[['buy_date', 'trading_days']].apply(
             lambda x: pd.to_datetime(x.buy_date)+timedelta(x.trading_days), 
             axis=1)
        br_df.sell_date = br_df.sell_date.astype(str)

        br_df['sell_close'] = round(br_df.buy_close*(1+br_df.pct_gain/100),2)

        cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
                'trading_days', 'daily_return', 'ticker']
        br_df = br_df.drop(['pct_gain', 'day_gain', self.ret_col], axis=1)[cols]

        idx = br_df.buy_date >= max(self.possible_trades_df.buy_date)
        br_df = br_df[idx]
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
        self.possible_trades_df = pd.read_csv(TRADE_FNM)
        self.buy_opportunities_df = pd.read_csv(BUY_FNM)

        # Augment buy recommendations
        self.ret_col = 'gain_daily_ret'
        self.augment_possible_trades_with_buy_opportunities()

        # Determine start and end date for backtesting period.
        self.start_date = min(self.possible_trades_df.buy_date)
        self.end_date   = max(self.possible_trades_df.sell_date)
        self.start_date = add_days_to_date(self.start_date, -5)
        self.end_date   = add_days_to_date(self.end_date, 5)

        # Pull down MSFT stock for period and use that as basis for determining
        # the stock market trading days
        success, hist = get_stock('MSFT', 2, "max")
        assert success == True, 'Failed to retrieve MSFT data'
        idx = (hist.index >= self.start_date) & (hist.index <= self.end_date)
        self.backtest_trading_dates = hist.loc[idx].index.to_list()
        self.start_date = min(self.backtest_trading_dates)
        self.end_date   = max(self.backtest_trading_dates)
    
    def pct_desired(self, threshold):
        # Read ticker statistics
        self.ticker_stats_df = pd.read_csv(STATS_FNM)
        idx = (self.ticker_stats_df.pct_desired >= threshold) \
            & (self.ticker_stats_df.daily_ret > 0)
        self.tickers = self.ticker_stats_df[idx].ticker.to_list()
        self.len_tickers = len(self.tickers)
        log(f"tickers={self.tickers} len_tickers={self.len_tickers}\n\n")

    def log_invested(self, message):
        log(message)
        log(f"invested={list(self.myPnL.invested.keys())}"
            f" ({len(self.myPnL.invested)})")

    def log_PnL(self, message):
        log(message)
        log(f"capital={self.myPnL.capital} in_use={self.myPnL.in_use}"
            f" free={self.myPnL.free}")
    
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
            f"{len(self.backtest_trading_dates)}\n", True)

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

    def find_lowest_expected_gain(self):
        lowest_expected_gain = None 
        lowest_ticker        = None

        for t in self.myPnL.invested.keys():
            tidx = self.ticker_stats_df.ticker == t
            t_total_days = float(self.ticker_stats_df.total_days.loc[tidx])
            t_total_cnt  = float(self.ticker_stats_df.total_cnt.loc[tidx])
            t_expected_days = t_total_days / t_total_cnt
            t_daily_ret = float(self.ticker_stats_df[self.ret_col].loc[tidx])
            t_expected_gain = (1 + (t_daily_ret/100)) ** t_expected_days - 1                       

            if lowest_expected_gain is None or \
               t_expected_gain < lowest_expected_gain:
                lowest_expected_gain = t_expected_gain
                lowest_ticker        = t

        return lowest_ticker, lowest_expected_gain

    def find_stock_to_replace(self):
        idx = self.ticker_stats_df.ticker == self.ticker
        total_days = float(self.ticker_stats_df.total_days.loc[idx])
        total_cnt  = float(self.ticker_stats_df.total_cnt.loc[idx])
        expected_days = total_days / total_cnt
        expected_gain = (1 + (self.daily_ret/100)) ** expected_days - 1

        lowest_ticker, lowest_expected_gain = self.find_lowest_expected_gain()

        if lowest_expected_gain is not None and \
           expected_gain > lowest_expected_gain:
            log(f"*** selling {lowest_ticker} on {self.trading_date} to free "
                f"up money for {self.ticker}")
            self.myPnL.sell_stock(lowest_ticker, self.trading_date)
        else:
            log(f"maxed out: {self.ticker} is not expected to perform better"
                f" than stocks already invested in")
            log(f"invested in: {list(self.myPnL.invested.keys())} "
                f"({len(self.myPnL.invested)})")
            log('') 

    def calc_amount(self):
        self.amount = self.myPnL.capital / self.max_stocks
        log(f"*** buying {self.amount} in {self.ticker} "
            f"on {self.buy_date} with target sell date of"
            f" {self.sell_date}")

    def buy_stock(self):
        log(f"enough money to buy {self.ticker}")
        self.log_invested("invested in:")
        self.log_PnL("before buy: myPnL=")
        self.myPnL.buy_stock(self.ticker, self.buy_date, 
                self.sell_date, self.amount)
        self.log_invested("after buy: invested in")
        self.log_PnL("after buy: myPnL=")

        # save the sell date for future processing
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
        else:
            log(f"not enough money to buy 25% of stock; not buying")
            self.log_invested(f"invested in:")
            self.log_PnL(f'myPnL=')

    def sell_stocks(self, to_sell):
        for ticker in to_sell:
            if ticker in self.myPnL.invested:
                self.log_invested(f"before selling {ticker}")
                self.log_PnL(f"before selling {ticker}")
                self.myPnL.sell_stock(ticker, self.trading_date)
                self.log_invested(f"after selling {ticker}")
                self.log_PnL(f"after selling {ticker}")

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
        cap_before = self.myPnL.capital
        log(f"before day_close {self.trading_date}:")
        self.log_invested(f"before day close {self.trading_date}:")
        self.log_PnL("before day close")
        tol = self.myPnL.capital - self.myPnL.in_use - self.myPnL.free
        tol = round(abs(tol), 6)
        log(f'tol={tol} tolerance < TOLERANCE={tol < TOLERANCE}')

        self.myPnL.day_close(self.trading_date)

        log(f"after day_close {self.trading_date}:")
        self.log_invested(f"after day close {self.trading_date}:")
        self.log_PnL("after day close")
        tol = self.myPnL.capital - self.myPnL.in_use - self.myPnL.free
        tol = round(abs(tol), 6)
        log(f'tol={tol} tolerance < TOLERANCE={tol < TOLERANCE}') 

    def main_back_test_loop(self):

        for self.trading_day, self.trading_date in \
            enumerate(tqdm(self.backtest_trading_dates, desc="simulate trades: ")):

            self.trading_date = str(self.trading_date)[:10]
            self.process_sell_signals()
            self.process_buy_opportunities()
            self.close_day()

    # TODO: streamline and correct
    def make_buy_recommendations(self):

        cols = ['ticker', 'trading_days', 'gain_pct', 'daily_return']
        mean_dict = self.possible_trades_df[cols].groupby('ticker').agg(['mean']).to_dict()
        mean_df = self.possible_trades_df[cols].groupby('ticker').agg(['mean']).reset_index()
        mean_df.columns=['ticker', 'trading_days', 'gain_pct', 'daily_return']
        self.buy_opportunities_df = pd.merge(self.buy_opportunities_df, mean_df, how='inner')

        # Get today's and yesterday's date
        today = datetime.today()
        yesterday = today - timedelta(1)
        today, yesterday = today.strftime('%Y-%m-%d'), yesterday.strftime('%Y-%m-%d')

        buy_opportunities_df = pd.merge(buy_opportunities_df, ticker_stats_df, how='inner')

        log('', True)
        log("Today's buying recommendations:\n", True)
        idx = (buy_opportunities_df.buy_date == today) #& (buy_opportunities_df.gain_pct > 0)
        df = buy_opportunities_df.loc[idx].sort_values(by=ret_col, ascending=False)[0:self.max_stocks]
        cols = ['ticker', 'buy_date', ret_col, 'gain_ratio', 'e_gain_daily_ret', 'e_loss_daily_ret',
                'day_gain', 'day_loss', 'day_zero']
        log(df[cols], True)
        log('', True)

        log('', True)
        log("Yesterday's buying recommendations:\n", True)
        idx = (buy_opportunities_df.buy_date == yesterday) & (buy_opportunities_df.gain_pct > 0)
        df = buy_opportunities_df.loc[idx].sort_values(by=ret_col, ascending=False)[0:self.max_stocks]
        log(df[cols], True)
        log('', True)

    def run_back_test(self, capital, max_stocks):
        self.init_back_test_run(capital, max_stocks)
        self.main_back_test_loop()
        self.myPnL.df.days_in_trade = self.myPnL.df.days_in_trade.astype(int)
        #self.make_buy_recommendations()

def main():
    bt = Backtester()

    capital_dict = {}
    invested_dict = {}
    len_tickers_dict = {}

    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for th in thresholds:
        bt.pct_desired(th)
        bt.run_back_test(10000, 5)
        log('', True)
        log(f'threshold={th}')
        bt.log_PnL('myPnL=')
        bt.log_invested('invested in:')

        # save key stats of run
        capital_dict[th]  = bt.myPnL.capital
        invested_dict[th] = bt.myPnL.invested.keys()
        len_tickers_dict[th]  = bt.len_tickers

        bt.myPnL.myCapital.df.index = bt.myPnL.myCapital.df.date
        to_plot_cols = ['capital', 'in_use']
        bt.myPnL.myCapital.df[to_plot_cols].plot(figsize=(18,10))
        plt.savefig(f'{DATAPATH}figure_{th}.png')
        #plt.show()

    log('')
    log(f'capital_dict={capital_dict}')
    log('')
    log(f'invested_dict={invested_dict}')
    log('')
    log(f'tickers_dict={len_tickers_dict}')
    log('')

    # Print summary...
    log('Threshold\tCapital\t\tReturn\tLen\tInvested', True)
    trading_days = 757
    for th in thresholds:
        cap = round(capital_dict[th],0)
        ret = ( (cap/10000) ** (1/trading_days)) -1
        ret = round (ret * 100, 2)
        inv = list(invested_dict[th])
        len = len_tickers_dict[th]
        log(f'{th}\t\t{cap}\t{ret}%\t{len}\t{inv}', True)

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
        main()