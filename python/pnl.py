import pandas                as pd
import numpy                 as np
from   util                  import log
import yfinance              as yf

# Constants
BUY       = 1
SELL      = 2
TOLERANCE = 1e-6
STOP_LOSS = -10 # max loss: -10%

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
        
        assert abs(capital - in_use - free) < TOLERANCE, "capital and in_use + free deviating too much!"
        
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
        self.capital    = capital
        self.in_use     = in_use
        self.free       = free
        self.max_stocks = max_stocks
        
    def buy_stock (self, ticker, buy_date, sell_date, amount):
        """
        Buy a stock on a specifid day and store it in the actual trades dataframe (i.e. df).
        
        The active investments are recorded in the invested dictionary. The historical
        stock data is stored in invested[ticker].

        Returns nothing.
        """
        
        assert amount > 0,                      f"amount ({amount}) needs to be greater than zero!"
        assert ticker not in self.invested,     f"already own shares in {ticker}!"
        assert len(self.invested) < self.max_stocks, f"already own maximum # stocks ({self.max_stocks})!"
        assert abs(self.capital - self.in_use - self.free) < TOLERANCE, "capital and in_use + free deviating too much!"
        
        # Make sure we have the money to buy stock
        if amount > self.free:
            if self.free > 0:
                amount=self.free
                log(f"you do not have {amount} and setting amount to {self.free}")
            else:
                log(f"you do not have any money left to buy ({self.free})! Not buying...")
                return

        # Retrieve the historical data for stock ticker and save it while we're invested
        asset  = yf.Ticker(ticker)
        hist   = asset.history(start=self.start, end=self.end)
        if len(hist) == 0:
            log(f'Failed to retrieve data from yf ticker={ticker} to buy stock')
            log('Skipping...')
            return

        idx = hist.index == buy_date
        if len(hist.Close.loc[idx]) != 1:
            log(f"PnL.buy_stock(): Unable to retrieve {buy_date} for {ticker}")
            log('Skipping...')
            return

        # Get share price and calculate how many shares we can buy
        # Also, set stop loss share price at 10 %
        self.invested[ticker] = hist.copy()
        idx = self.invested[ticker].index == buy_date
        share_price = float(self.invested[ticker].Close.loc[idx])
        no_shares = amount / share_price
        stop_loss = share_price * 0.9
        
        # Reduce free and increase in_use by amount
        self.free   = self.free - amount
        self.in_use = self.in_use + amount
        assert abs(self.capital - self.in_use - self.free) < TOLERANCE, \
            "capital and in_use + free deviating too much!"
        
        # Store the buy action in self.df data frame
        buy_dict = {'date'         : [buy_date],
                    'ticker'       : [ticker],
                    'action'       : ['BUY'],
                    'orig_amount'  : [amount],
                    'close_amount' : [amount],
                    'no_shares'    : [no_shares],
                    'stop_loss'    : [stop_loss],
                    'daily_gain'   : [0.0],
                    'daily_pct'    : [0.0],
                    'days_in_trade': [0],
                    'invested'     : [1]
                   }
        
        buy_df = pd.DataFrame(buy_dict)
        self.df = pd.concat([self.df, buy_df])
        self.df.invested = self.df.invested.astype(int)
     
    def sell_stock (self, ticker, sell_date):
        """
        Sell stock on specified date. Also, remove the ticker from invested after
        the position has been closed. 

        Returns nothing.
        """

        assert self.capital >= 0, "capital needs to be zero or greater"
        assert self.in_use  >= 0, "in_use needs to be zero or greater"
        assert self.free    >= 0, "free needs to be zero or greater"        
        assert abs(self.capital - self.in_use - self.free) < TOLERANCE, \
               "capital and in_use + free deviating too much!"

        # Return if we do not own the stock (may be due to a forced stop-loss sales)
        if ticker not in self.invested:
            return 
        
        # Get the latest close_amount for ticker and no_shares owned
        idx           = (self.df.ticker == ticker) & (self.df.invested==1)
        no_shares     = float(self.df['no_shares'].loc[idx])
        close_amount  = float(self.df['close_amount'].loc[idx])
        orig_amount   = float(self.df['orig_amount'].loc[idx])
        stop_loss     = float(self.df['stop_loss'].loc[idx])
        days_in_trade = int(self.df['days_in_trade'].loc[idx])
        self.df.loc[idx, 'invested'] = 0
        
        # Calculate how much the sell will earn
        idx = self.invested[ticker].index == sell_date
        if len(self.invested[ticker].Close.loc[idx]) != 1:
            return
        share_price   = float(self.invested[ticker].Close.loc[idx])
        today_amount  = no_shares * share_price
        delta_amount  = today_amount - close_amount
        delta_pct     = (delta_amount / close_amount) * 100

        # print the profit/loss of the trade
        log(f"profit of selling {ticker} on {sell_date}: "
              f"{today_amount - orig_amount}"
              f"{round(((today_amount - orig_amount)/orig_amount)*100,2)}%")
        
        # Correct in_use and capital for delta_amount
        self.capital  = self.capital + delta_amount
        self.in_use   = self.in_use  + delta_amount
        
        # Shift today's amount (in_use -> free)
        # We do not allow in_use to become negative, even if it is by
        # a small amount...
        self.in_use   = self.in_use - today_amount
        if self.in_use < 0:
            self.in_use = 0 
        self.free     = self.free   + today_amount

        if abs(self.capital - self.in_use - self.free) > TOLERANCE:
            log("self.capital=", self.capital)
            log("self.in_use=", self.in_use)
            log("self.free=", self.free)
            log("diff=", abs(self.capital - self.in_use - self.free))
            assert abs(self.capital - self.in_use - self.free) < TOLERANCE, \
                   "capital and in_use + free deviating too much!"
        
        # Save the stock sell
        sell_dict = {'date'        : [sell_date],
                    'ticker'       : [ticker],
                    'action'       : ['SELL'],
                    'orig_amount'  : [orig_amount],
                    'close_amount' : [today_amount],
                    'no_shares'    : [no_shares],
                    'stop_loss'    : [stop_loss],
                    'daily_gain'   : [delta_amount],
                    'daily_pct'    : [delta_pct],
                    'days_in_trade': [days_in_trade + 1],
                    'invested'     : [ 0 ]
                   }
        
        sell_df = pd.DataFrame(sell_dict)
        self.df = pd.concat([self.df, sell_df])

        # Remove stock from invested dictionary
        del self.invested[ticker]
        
    def day_close(self, close_date):
        """
        Day end close. Updates the value of the stocks actively invested in and
        updates the variables capital, in_use, and free. After the value of each
        position has been updated, store capital, in_use, and free.

        It also has a safety net to print a warning if the value of an open 
        position changes by more than ten percent. This is put in place as 
        occasionally the data returned by yfinance is incorrect. This will alert
        us that this may be happening. 

        Returns nothing.
        """
        # print("day_close:")
        tickers = list(self.invested.keys())
        for ticker in tickers:
            
            # Get the latest close_amount for ticker and no_shares owned
            df_idx        = (self.df.ticker == ticker) & (self.df.invested==1)
            if len(self.df.loc[df_idx]) == 0:
                continue
            log(f"{ticker}:\n {self.df.loc[df_idx]}")
            no_shares     = float(self.df['no_shares'].loc[df_idx])
            close_amount  = float(self.df['close_amount'].loc[df_idx])
            orig_amount   = float(self.df['orig_amount'].loc[df_idx])
            stop_loss     = float(self.df['stop_loss'].loc[df_idx])
            days_in_trade = int(self.df['days_in_trade'].loc[df_idx])
            #log(self.df.info())
            self.df.loc[df_idx, 'invested'] = 0

            # Calculate how much the sell will earn
            hist_idx      = self.invested[ticker].index == close_date
            if len(self.invested[ticker].Close.loc[hist_idx]) == 0:
                continue

            share_price   = float(self.invested[ticker].Close.loc[hist_idx])
            today_amount  = no_shares * share_price
            delta_amount  = today_amount - close_amount
            delta_pct     = (delta_amount / close_amount) * 100

            # check if we reached a stop loss condition
            gain_pct = ((today_amount - orig_amount) / orig_amount) * 100
            if share_price < stop_loss:
                log(f"breached stop-loss and selling {ticker}...")
                self.df.loc[df_idx, 'invested'] = 1
                self.sell_stock(ticker, close_date)
                continue

            # Report a suspicious high change per stock/day. Threshold for now set at 10%
            # Allows us to see what other stocks may have issues than just SBT...
            if abs(delta_amount / self.capital) > 0.1:
                log('', True)
                log('********************', True)
                log(f'*** WARNING      *** capital changed by more than 10% for {ticker} on {close_date}!', True)
                log(f'***              *** no_shares={no_shares} share_price={share_price} today_amount={today_amount}', True)
                log(f'***              *** orig_amount={orig_amount} close_amount={close_amount} delta_amount={delta_amount}', True)
                log('********************', True)
                log('', True)
            
            # Correct in_use and capital for delta_amount
            self.capital  = self.capital + delta_amount
            self.in_use   = self.in_use  + delta_amount
            assert abs(self.capital - self.in_use - self.free) < TOLERANCE, \
                   "capital and in_use + free deviating too much!"

            close_dict = {'date'        : [close_date],
                         'ticker'       : [ticker],
                         'action'       : ['CLOSE'],
                         'orig_amount'  : [orig_amount],
                         'close_amount' : [today_amount],
                         'no_shares'    : [no_shares],
                         'stop_loss'    : [stop_loss],
                         'daily_gain'   : [delta_amount],
                         'daily_pct'    : [delta_pct],
                         'days_in_trade': [days_in_trade + 1],
                         'invested'     : [ 1 ]
                   }
        
            close_df = pd.DataFrame(close_dict)
            self.df = pd.concat([self.df, close_df])
            
        # Store overall end day result in myCapital
        self.myCapital.day_close(close_date, self.capital, self.in_use, self.free)
