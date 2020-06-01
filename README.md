# Stockie

A stock trading system that analyzes 2700+ stocks to identify those stocks that have a high probability to 
rise in the next few days. By buying and selling these stocks, Stockie is able to turn a $10K investment 
into $150K - $200K over a three-year period. Much better than market returns in a good year which is around 
20 – 30 percent per annum.

## Prerequisites

Stockie requires the following to be installed:

Python 3.7
Scikit-Learn 0.23
yfinance 0.1.54

## How to Install

1) Clone and down load Stockie repository to your machine using

$ git clone https://github.com/frkornet/Stockie

2) Open symbols.py in the python subdirectory in an editor and change "/Users/frkornet/Stockie/" to point 
to the directory where you cloned Stockie

- LOGPATH         = '<Stockie root directory>/log/'
- DATAPATH        = '<Stockie root directory>/data/'
- PICPATH         = '<Stockie root directory>/pic/'
- MODELPATH       = '<Stockie root directory>/model/'

## How to Use

You can run Stockie by calling job.py

$ python job.py

That will run the python programs in the right order and will create a set of output files:
- qa.py: identify the stocks that fail to smooth a curve through the close stock prices
- trade.py: build possible buy and sell pairs
- stats.py: augment the possible buy and sell pairs with the statistics for each stock at that point in time ensuring that we do not leak information
- backtest.py: for each set of parameters run a backtest

The program will create a log file in the "log" sub-directory (prefixed with "job" followed by date and time when job started.

You can specify the combinations that you want to run by editing backtest.py:

thresholds =[
            (True,  0.05, "0m", 0, "daily_ret"      ),
            (False, 0.05, "0m", 0, "daily_ret"      ),

            (True,  0.05, "0m", 10, "daily_ret"     ),
            (False, 0.05, "0m", 10, "daily_ret"     ),

	    ...

            (True,  0.05, "0m", 40, "daily_ret"      ),
            (False, 0.05, "0m", 40, "daily_ret"      ),
        ]
        
The first param is whether or not the statistics need to be cointinually updated while the backtest runs. 
The second parameter is batchsize (percentage). 
The third parameter is how many months to keep data for. By specifying "0m" all data is kept when updating 
the statistics at the end of each backtested trading day. 
The fourth parameter states the threshold for pct_increase (heuristic rule). 
The fifth parameter specifies which return value to use. There are two possibilities: "gain_daily_ret" and "daily_ret".

The actual trades executed for a backtest run are stored in "actual_<threshold>.csv" in data. 
The graphical chart showing the growth of capital over time are stored in "trade_threshold_<threshold>.png" 
in the "pic" sub-directory.
  
## How to Contribute

Let me know if you want to contribute to Stockie and I will agree a way we can work together.

## Add Contributors

None at this point.

## Acknowledgements

Bryan Arnolds from Flatiron for the initial idea of smoothing a curve. Also, Carson Lloyd for developing initial web scraping.

## Contact Information

If you want to contact the developer, please send me an email to frkornet@gmail.com addressed to "Frank Kornet" with subject "Stockie". I will read my email daily and will respond ones I get your email.
