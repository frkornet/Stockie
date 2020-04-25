import pandas                as pd
import numpy                 as np
from   util                  import log, add_days_to_date, get_stock_start
from   symbols               import BUY, SELL, TOLERANCE, STOP_LOSS
import yfinance              as yf
import gc; gc.enable()

class Capital(object):
    """
    A simple class to record the growth of decline of capital, in_use, and 
    free after each trading day. The data is stored in a dataframe with
    four columns: date, capital, in_use, and free. The method day_close()
    is used to store the capital, in_use, and free after closing the day.

    This class is used by PnL class whgich is the real workhorse. 
    """
    def __init__(self):
        cols = [ 'date', 'capital', 'in_use', 'free']
        self.df = pd.DataFrame(columns=cols)
        
    def day_close (self, close_date, capital, in_use, free):

        assert capital >= 0, "capital needs to be zero or greater"
        assert in_use  >= 0, "in_use needs to be zero or greater"
        assert free    >= 0, "free needs to be zero or greater"
        
        assert abs(capital - in_use - free) < TOLERANCE, \
            "capital and in_use + free deviating too much!"
        
        close_dict = { 'date'    : [close_date], 
                       'capital' : [capital],
                       'in_use'  : [in_use],
                       'free'    : [free]
                     }
        
        self.df = pd.concat( [self.df, pd.DataFrame(close_dict)] )

class PnL(object):
    """
    The PnL class keeps track of all the buys, sells, and day closes. 
    The class keeps track of things via a dataframe. The dataframe consists 
    of the following columns: date, ticker, action (BUY, SELL, or CLOSE), 
    original amount invested in the stock, close amount at the end of the 
    trading day, number of shares owned, the stop loss price undert which we 
    will sell the stock (typically 90% of the starting share price),
    the daily gain (positive is gain; negative is loss), daily compounded
    return percentage, and a flag invested.

    The flag is used to retrieve the last record for a particular active 
    trade. The class ensures that there is always one record for which 
    invested is 1. It is important to ensure this is the case as otherwise 
    the code will not work.

    The class consists of three methods:

    - buy_stock() : buy a specified amount of a particular stock on buy date
                    with a planned sell date
    - sell_stock(): sell a stock on a specific sell date
    - day_close() : for each open ticker investment update the value of the
                    position. If the share price droipped below stop loss,
                    carry out a forced sell of stock. After that
                    record the day end capital, in_use, and free.
    
    The class enforces that no more than max_stocks are owned. max_stocks is 
    set at initialization. It also enforces, that capital = in_use + free. 
    Since these are floats, the code use the following trick to ensure they 
    are basically the same: abs(capital - in_use - free) < TOLERANCE where
    TOLERANCE is 1E-6 (i.e. close to zero).

    """

    def __init__(self, start_date, end_date, capital, in_use, free, max_stocks):
        
        cols = [ 'date', 'ticker', 'action', 'orig_amount', 'close_amount',
                 'no_shares', 'stop_loss', 'daily_gain', 'daily_pct', 
                 'days_in_trade', 'invested']
        self.df = pd.DataFrame(columns=cols)

        self.myCapital  = Capital()   
        self.invested   = {}
        self.start      = start_date
        self.end        = end_date
        self.end_plus_1 = add_days_to_date(end_date, 1)
        self.capital    = capital
        self.in_use     = in_use
        self.free       = free
        self.max_stocks = max_stocks
    
    def validate_buy(self):
        msg = f"amount ({self.amount}) needs to be greater than zero!"
        assert self.amount > 0, msg
        
        msg = f"already own shares in {self.ticker}!"
        assert self.ticker not in self.invested, msg
        
        msg = f"already own maximum # stocks ({self.max_stocks})!"
        assert len(self.invested) < self.max_stocks, msg
            
        tol = abs(self.capital - self.in_use - self.free)
        assert tol < TOLERANCE, f"tolerance {tol} deviating too much!"

    def check_sufficient_amount(self):
        if self.amount > self.free:
            if self.free > 0:
                self.amount=self.free     # clipping amount
            else:
                return False                   # not buying
        return True

    def buy_stock_init(self, ticker, buy_date, sell_date, amount):
        self.ticker = ticker
        self.buy_date = buy_date
        self.sell_date = sell_date
        self.amount = amount
        self.validate_buy()
        return self.check_sufficient_amount()

    def get_hist(self):
        s, e = self.start, self.end_plus_1
        success, hist = get_stock_start(self.ticker, 2, s, e)
        if success == False:
            log(f'Failed to retrieve data from yf ({ticker}) to buy stock')
            return False

        idx = hist.index == self.buy_date
        if len(hist.Close.loc[idx]) != 1:
            log(f"Unable to retrieve {buy_date} for {ticker}")
            return False
        
        self.hist = hist
        return True
    
    def calc_no_shares_to_buy(self):
        self.invested[self.ticker] = self.hist.copy()
        idx = self.invested[self.ticker].index == self.buy_date
        self.share_price = float(self.invested[self.ticker].Close.loc[idx])
        self.no_shares = self.amount / self.share_price
        self.stop_loss = self.share_price * 0.9

    def update_buy_amount(self):
        self.free   = self.free - self.amount
        self.in_use = self.in_use + self.amount
        tol = abs(self.capital - self.in_use - self.free)
        assert tol < TOLERANCE, f"tolerance ({tol}) deviating too much!"

    def save_buy(self):
        buy_dict = {'date'         : [self.buy_date],
                    'ticker'       : [self.ticker],
                    'action'       : ['BUY'],
                    'orig_amount'  : [self.amount],
                    'close_amount' : [self.amount],
                    'no_shares'    : [self.no_shares],
                    'stop_loss'    : [self.stop_loss],
                    'daily_gain'   : [0.0],
                    'daily_pct'    : [0.0],
                    'days_in_trade': [0],
                    'invested'     : [1]
                   }
        
        buy_df = pd.DataFrame(buy_dict)
        self.df = pd.concat([self.df, buy_df])
        self.df.invested = self.df.invested.astype(int)   

    def buy_stock (self, ticker, buy_date, sell_date, amount):
        """
        Buy a stock on a specifid day and store it in the actual trades 
        dataframe (i.e. df).
        
        The active investments are recorded in the invested dictionary. 
        The historical stock data is stored in invested[ticker].

        Returns nothing.
        """

        if self.buy_stock_init(ticker, buy_date, sell_date, amount) == False:
            return

        if self.get_hist() == False:
            return

        self.calc_no_shares_to_buy()
        self.update_buy_amount()  
        self.save_buy()
    
    def validate_sell(self):
        assert self.capital >= 0, "capital needs to be zero or greater"
        assert self.in_use  >= 0, "in_use needs to be zero or greater"
        assert self.free    >= 0, "free needs to be zero or greater"
        tol = abs(self.capital - self.in_use - self.free) 
        assert tol < TOLERANCE, "tolerance {tol} deviating too much!"

    def get_sell_index(self, ticker):
        self.ticker = ticker
        idx    = (self.df.ticker == self.ticker) & (self.df.invested==1)
        len_idx = len(self.df[idx])
        assert len_idx == 1, "Did not get latest record index"
        self.idx = idx

    def get_last_record(self, ticker):

        self.get_sell_index(ticker)
        self.no_shares     = float(self.df['no_shares'].loc[self.idx])
        self.close_amount  = float(self.df['close_amount'].loc[self.idx])
        self.orig_amount   = float(self.df['orig_amount'].loc[self.idx])
        self.stop_loss     = float(self.df['stop_loss'].loc[self.idx])
        self.days_in_trade = int(self.df['days_in_trade'].loc[self.idx])

    def get_hist_sell_date(self, sell_date):
        self.sell_date = sell_date
        idx = self.invested[self.ticker].index == sell_date
        len_idx = len(self.invested[self.ticker].Close.loc[idx])
        assert len_idx == 1, "Did not get {sell_date} record index"
        self.idx = idx

    def calc_profit_from_sales(self):
        self.share_price=float(self.invested[self.ticker].Close.loc[self.idx])
        self.share_price   = max(self.stop_loss, self.share_price)
        self.today_amount  = self.no_shares * self.share_price
        self.delta_amount  = self.today_amount - self.close_amount
        self.delta_pct     = (self.delta_amount / self.close_amount) * 100
    
    def get_sell_share_price(self, ticker, sell_date):
        self.get_last_record(ticker)
        self.get_hist_sell_date(sell_date)


    def update_sell_delta_amount(self):
        self.capital  = self.capital    + self.delta_amount
        self.in_use   = self.in_use     + self.delta_amount
        
        self.in_use   = max(self.in_use - self.today_amount, 0.0)
        self.free     = self.free       + self.today_amount

        self.validate_sell()

        
    def save_sell(self):
        idx = self.df.ticker == self.ticker
        self.df.loc[idx, 'invested'] = 0

        sell_dict = {'date'        : [self.sell_date],
                    'ticker'       : [self.ticker],
                    'action'       : ['SELL'],
                    'orig_amount'  : [self.orig_amount],
                    'close_amount' : [self.today_amount],
                    'no_shares'    : [self.no_shares],
                    'stop_loss'    : [self.stop_loss],
                    'daily_gain'   : [self.delta_amount],
                    'daily_pct'    : [self.delta_pct],
                    'days_in_trade': [self.days_in_trade + 1],
                    'invested'     : [ 0 ]
                   }
        
        sell_df = pd.DataFrame(sell_dict)
        self.df = pd.concat([self.df, sell_df])

    def sell_stock (self, ticker, sell_date):
        """
        Sell stock on specified date. Also, remove the ticker from invested 
        after the position has been closed. 

        Returns nothing.
        """
        
        self.validate_sell()
        if ticker not in self.invested:
            return 
        
        self.get_sell_share_price(ticker, sell_date)
        self.calc_profit_from_sales()    
        self.update_sell_delta_amount()
        self.save_sell()

        del self.invested[ticker]
    
    def get_latest_index(self, ticker):
        idx  = (self.df.ticker == ticker) & (self.df.invested==1)
        len_idx = len(self.df.loc[idx])
        assert len_idx == 1, f"len_idx must be 1 (={len_idx}"
        self.idx = idx

    def extract_latest_info(self, ticker):
        self.get_latest_index(ticker)
        self.ticker        = ticker
        self.no_shares     = float(self.df['no_shares'].loc[self.idx])
        self.close_amount  = float(self.df['close_amount'].loc[self.idx])
        self.orig_amount   = float(self.df['orig_amount'].loc[self.idx])
        self.stop_loss     = float(self.df['stop_loss'].loc[self.idx])
        self.days_in_trade = int(self.df['days_in_trade'].loc[self.idx])

    def get_shareprice(self, ticker):
        self.extract_latest_info(ticker)
        hist_idx = self.invested[ticker].index == self.close_date
        hist_len = len(self.invested[ticker].Close.loc[hist_idx])
        if hist_len == 0:
            return False

        # max() is used in case there are multiple rows
        dividend = float(self.invested[ticker].Dividends.loc[hist_idx].max())
        share_price = float(self.invested[ticker].Close.loc[hist_idx].max())

        self.dividend = 0
        self.share_price = share_price
        if dividend > 0:
            self.dividend = dividend

        return True

    def calc_delta_amount(self):
        self.today_amount  = self.no_shares * self.share_price
        self.delta_amount  = self.today_amount - self.close_amount
        self.delta_pct     = (self.delta_amount / self.close_amount) * 100
        self.gain_pct      = ((self.today_amount - self.orig_amount) \
                           / self.orig_amount) * 100

        if abs(self.delta_amount / self.capital) > 0.1:
            self.print_warning()

    def print_warning(self):
        log('')
        log('********************')
        log(f'*** WARNING      *** capital changed by more than 10%'
            f' for {self.ticker} on {self.close_date}!')
        log(f'***              *** no_shares={self.no_shares} '
            f'share_price={self.share_price} today_amount={self.today_amount}')
        log(f'***              *** orig_amount={self.orig_amount} '
            f'close_amount={self.close_amount} delta_amount={self.delta_amount}')
        log('********************')
        log('')

    def update_delta_amount(self):
        self.capital  = self.capital + self.delta_amount
        self.in_use   = self.in_use  + self.delta_amount
        tol = abs(self.capital - self.in_use - self.free)
        assert tol < TOLERANCE, "tol deviating too much!"

    def add_close_record(self):
        idx = self.df.ticker == self.ticker
        self.df.loc[idx, 'invested'] = 0

        close_dict = {'date'       : [self.close_date],
                    'ticker'       : [self.ticker],
                    'action'       : ['CLOSE'],
                    'orig_amount'  : [self.orig_amount],
                    'close_amount' : [self.today_amount],
                    'no_shares'    : [self.no_shares],
                    'stop_loss'    : [self.stop_loss],
                    'daily_gain'   : [self.delta_amount],
                    'daily_pct'    : [self.delta_pct],
                    'days_in_trade': [self.days_in_trade + 1],
                    'invested'     : [ 1 ]
                }
    
        close_df = pd.DataFrame(close_dict)
        self.df = pd.concat([self.df, close_df])

    def process_dividend(self):
        if self.dividend == 0:
            return

        dividend_amount = self.dividend * self.no_shares
        self.capital = self.capital + dividend_amount
        self.free = self.free + dividend_amount
        tol = abs(self.capital - self.in_use - self.free)
        assert tol < TOLERANCE, "tol deviating too much!"

    def update_dicts(self):
        self.no_shares_dict[self.ticker] = self.no_shares
        self.share_price_dict[self.ticker] = self.share_price

    def day_close_ticker(self, ticker):
        if self.get_shareprice(ticker) == False: 
            return

        if self.share_price < self.stop_loss:
            self.sell_stock(ticker, self.close_date)
            return

        self.calc_delta_amount()          
        self.update_delta_amount()
        self.process_dividend()
        self.add_close_record()
        self.update_dicts()

    def init_day_close(self, close_date):
        self.tickers = list(self.invested.keys())
        self.close_date = close_date
        self.no_shares_dict = {}
        self.share_price_dict = {}

    def is_rebalance_needed(self):
        max_value = (self.capital / self.max_stocks) * 2
        tickers = list(self.invested.keys())
        for t in tickers:
            value = self.no_shares_dict[t] * self.share_price_dict[t]
            if value > max_value:
                return True
        return False

    def rebalance_if_needed(self):
        if self.is_rebalance_needed() == True:
            log(f'*** rebalance needed at {self.close_date}')


    def day_close(self, close_date):
        """
        Day end close. 
        """

        gc.collect()
        self.init_day_close(close_date)
        for ticker in self.tickers:
            self.day_close_ticker(ticker)

        self.rebalance_if_needed()
        self.myCapital.day_close(self.close_date, self.capital, self.in_use, 
                                 self.free)
        gc.collect()
