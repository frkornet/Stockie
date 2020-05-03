
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

TRADE_COLS      = [ 'buy_date', 'buy_close', 'sell_date', 'sell_close',
                    'gain_pct', 'trading_days', 'daily_return', 'ticker' ]
TRADE_COL_TYPES = [ str, float, str, float, float, int, float, str ]

STAT_COLS       = ['ticker',
                   'cnt_gain', 'min_pct_gain', 'max_pct_gain', 'std_pct_gain',
                   'mean_pct_gain',  'mean_day_gain', 'gain_daily_ret',
                   'cnt_loss', 'min_pct_loss', 'max_pct_loss', 'std_pct_loss',
                   'mean_pct_loss',  'mean_day_loss', 'loss_daily_ret',
                   'cnt_zero',  'mean_day_zero',
                   'total_cnt', 'total_days', 'mean_day', 'daily_ret', 
                   'pct_desired', 'gain_ratio', 'good'
                ]

STAT_COL_TYPES  = [str, 
                   int, float, float, float, float, float, float,
                   int, float, float, float, float, float, float,
                   int, float,
                   int, int, float, float, float, float, int
                   ]
STAT_BATCH_SIZE = 0.05

BT_DRET_COL     = "gain_daily_ret"
