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
        
        tol = abs(capital - in_use - free)
        assert tol < TOLERANCE, f"tolerance {tol} deviating too much!"
        
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
    TOLERANCE is 1E-3 (i.e. close to zero).

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
    
    ###################################
    # Private methods for buy_stock() #
    ###################################

    def __validate_buy__(self):
        msg = f"amount ({self.amount}) needs to be greater than zero!"
        assert self.amount > 0, msg
        
        msg = f"already own shares in {self.ticker}!"
        assert self.ticker not in self.invested, msg
        
        msg = f"already own maximum # stocks ({self.max_stocks})!"
        assert len(self.invested) < self.max_stocks, msg
            
        tol = abs(self.capital - self.in_use - self.free)
        assert tol < TOLERANCE, f"tolerance {tol} deviating too much!"

    def __check_sufficient_amount__(self):
        if self.amount > self.free:
            if self.free > 0:
                self.amount=self.free     # clipping amount
            else:
                return False                   # not buying
        return True

    def __buy_stock_init__(self, ticker, buy_date, sell_date, amount):
        self.ticker = ticker
        self.buy_date = buy_date
        self.sell_date = sell_date
        self.amount = amount
        self.__validate_buy__()
        return self.__check_sufficient_amount__()

    def __get_hist__(self):
        s, e = self.start, self.end_plus_1
        success, hist = get_stock_start(self.ticker, 3, s, e)
        if success == False:
            log(f'Failed to retrieve {self.ticker} data to buy stock')
            return False

        idx = hist.index == self.buy_date
        len_idx = len(hist.loc[idx])
        if len_idx != 1:
            msg = f"Unable to retrieve {self.buy_date} for {self.ticker}" \
                + f" (len_idx={len_idx})"
            log(msg)
            return False
        
        self.hist = hist
        return True
    
    def __calc_no_shares_to_buy__(self):
        self.invested[self.ticker] = self.hist.copy()
        idx = self.invested[self.ticker].index == self.buy_date
        self.share_price = float(self.invested[self.ticker].Close.loc[idx])
        self.no_shares = self.amount / self.share_price
        self.stop_loss = self.share_price * 0.9

    def __update_buy_amount__(self):
        self.free   = self.free - self.amount
        self.in_use = self.in_use + self.amount
        tol = abs(self.capital - self.in_use - self.free)
        assert tol < TOLERANCE, f"tolerance ({tol}) deviating too much!"

    def __save_buy__(self):
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

        if self.__buy_stock_init__(ticker, buy_date, sell_date, amount) == False:
            return

        if self.__get_hist__() == False:
            return

        self.__calc_no_shares_to_buy__()
        self.__update_buy_amount__()  
        self.__save_buy__()
    
    ####################################
    # Private methods for sell_stock() #
    ####################################

    def __validate_sell__(self):
        assert self.capital >= 0, "capital needs to be zero or greater"
        assert self.in_use  >= 0, "in_use needs to be zero or greater"
        assert self.free    >= 0, "free needs to be zero or greater"
        tol = abs(self.capital - self.in_use - self.free) 
        assert tol < TOLERANCE, "tolerance {tol} deviating too much!"

    def __get_sell_index__(self, ticker):
        self.ticker = ticker
        idx    = (self.df.ticker == self.ticker) & (self.df.invested==1)
        len_idx = len(self.df[idx])
        assert len_idx == 1, "Did not get latest record index"
        self.idx = idx

    def __get_last_record__(self, ticker):

        self.__get_sell_index__(ticker)
        self.no_shares     = float(self.df['no_shares'].loc[self.idx])
        self.close_amount  = float(self.df['close_amount'].loc[self.idx])
        self.orig_amount   = float(self.df['orig_amount'].loc[self.idx])
        self.stop_loss     = float(self.df['stop_loss'].loc[self.idx])
        self.days_in_trade = int(self.df['days_in_trade'].loc[self.idx])

    def __get_hist_sell_date__(self, sell_date):
        self.sell_date = sell_date
        idx = self.invested[self.ticker].index == sell_date
        len_idx = len(self.invested[self.ticker].Close.loc[idx])
        assert len_idx == 1, "Did not get {sell_date} record index"
        self.idx = idx

    def __calc_profit_from_sales__(self):
        self.share_price=float(self.invested[self.ticker].Close.loc[self.idx])
        self.share_price   = max(self.stop_loss, self.share_price)
        self.today_amount  = self.no_shares * self.share_price
        self.delta_amount  = self.today_amount - self.close_amount
        self.delta_pct     = (self.delta_amount / self.close_amount) * 100
    
    def __get_sell_share_price__(self, ticker, sell_date):
        self.__get_last_record__(ticker)
        self.__get_hist_sell_date__(sell_date)


    def __update_sell_delta_amount__(self):
        self.capital  = self.capital    + self.delta_amount
        self.in_use   = self.in_use     + self.delta_amount
        
        self.in_use   = max(self.in_use - self.today_amount, 0.0)
        self.free     = self.free       + self.today_amount

        self.__validate_sell__()

        
    def __save_sell__(self):
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
        
        self.__validate_sell__()       
        self.__get_sell_share_price__(ticker, sell_date)
        self.__calc_profit_from_sales__()    
        self.__update_sell_delta_amount__()
        self.__save_sell__()

        del self.invested[ticker]
    
    ###################################
    # Private methods for day_close() #
    ###################################

    def __get_latest_index__(self, ticker):
        idx  = (self.df.ticker == ticker) & (self.df.invested==1)
        len_idx = len(self.df.loc[idx])
        assert len_idx == 1, f"len_idx must be 1 (={len_idx}"
        self.idx = idx

    def __extract_latest_info__(self, ticker):
        self.__get_latest_index__(ticker)
        self.ticker        = ticker
        self.no_shares     = float(self.df['no_shares'].loc[self.idx])
        self.close_amount  = float(self.df['close_amount'].loc[self.idx])
        self.orig_amount   = float(self.df['orig_amount'].loc[self.idx])
        self.stop_loss     = float(self.df['stop_loss'].loc[self.idx])
        self.days_in_trade = int(self.df['days_in_trade'].loc[self.idx])

    def __get_shareprice__(self, ticker):
        self.__extract_latest_info__(ticker)
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

    def __calc_delta_amount__(self):
        self.today_amount  = self.no_shares * self.share_price
        self.delta_amount  = self.today_amount - self.close_amount
        self.delta_pct     = (self.delta_amount / self.close_amount) * 100
        self.gain_pct      = ((self.today_amount - self.orig_amount) \
                           / self.orig_amount) * 100

        if abs(self.delta_amount / self.capital) > 0.1:
            self.__print_warning__()

    def __print_warning__(self):
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

    def __update_delta_amount__(self):
        self.capital  = self.capital + self.delta_amount
        self.in_use   = self.in_use  + self.delta_amount
        tol = abs(self.capital - self.in_use - self.free)
        assert tol < TOLERANCE, "tol deviating too much!"

    def __add_close_record__(self):
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

    def __process_dividend__(self):
        if self.dividend == 0:
            return

        dividend_amount = self.dividend * self.no_shares
        self.capital = self.capital + dividend_amount
        self.free = self.free + dividend_amount
        tol = abs(self.capital - self.in_use - self.free)
        assert tol < TOLERANCE, "tol deviating too much!"

    def __update_dicts__(self):
        self.no_shares_dict[self.ticker] = self.no_shares
        self.share_price_dict[self.ticker] = self.share_price

    def __day_close_ticker__(self, ticker):
        if self.__get_shareprice__(ticker) == False: 
            return

        if self.share_price < self.stop_loss:
            self.sell_stock(ticker, self.close_date)
            return

        self.__calc_delta_amount__()          
        self.__update_delta_amount__()
        self.__process_dividend__()
        self.__add_close_record__()
        self.__update_dicts__()

    def __init_day_close__(self, close_date):
        self.tickers = list(self.invested.keys())
        self.close_date = close_date
        self.no_shares_dict = {}
        self.share_price_dict = {}

    def __is_rebalance_needed__(self):
        max_value = (self.capital / self.max_stocks) * 2
        tickers = list(self.invested.keys())
        for t in tickers:
            value = self.no_shares_dict[t] * self.share_price_dict[t]
            if value > max_value:
                return True
        return False

    def __rebalance_if_needed__(self):
        if self.__is_rebalance_needed__() == True:
            log(f'*** rebalance needed at {self.close_date}')

    def day_close(self, close_date):
        """
        Day end close. 
        """

        gc.collect()
        self.__init_day_close__(close_date)
        for ticker in self.tickers:
            self.__day_close_ticker__(ticker)

        self.__rebalance_if_needed__()
        self.myCapital.day_close(self.close_date, self.capital, self.in_use, 
                                 self.free)
        gc.collect()
