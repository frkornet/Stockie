
LOGPATH         = '/Users/frkornet/Stockie/log/'
DATAPATH        = '/Users/frkornet/Stockie/data/'
PICPATH         = '/Users/frkornet/Stockie/pic/'
STOCKS_FNM      = f'{DATAPATH}stocks202002.csv'
EXCLUDE_FNM     = f'{DATAPATH}exclude.csv'
FULL_TRADE_FNM  = f'{DATAPATH}full_possible_trades.csv'
TRAIN_TRADE_FNM = f'{DATAPATH}train_possible_trades.csv'
TEST_TRADE_FNM  = f'{DATAPATH}test_possible_trades.csv'
STATS_FNM       = f'{DATAPATH}ticker_stats.csv'
BUY_FNM         = f'{DATAPATH}possible_buys.csv'
EXCLUDE_SET     = {'AGE', 'AMK', 'BURG', 'CFB', 'LBC', 'MEC', 'OSW', 'PSN', 
                   'PTI', 'SBT'}
TRADE_PERIOD    = "10y"
TRADE_THRESHOLD = 0.5
QA_PERIOD       = "15y"
TRADE_DAILY_RET = 50.0
YF_TIMES        = 3
SLEEP_TIME      = 5
QA_SET          = 1
TOLERANCE       = 1e-3
BUY             = 1
SELL            = 2
STOP_LOSS       = -10 # max loss: -10%
