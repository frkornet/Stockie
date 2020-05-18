import pandas as pd
import numpy as np

from   stats import Stats
from   util  import log_filename, log, empty_dataframe, is_holiday, \
                    get_current_day_and_time, calc_runtime, \
                    save_dataframe_to_csv
from symbols import TRADE_COLS, TRADE_COL_TYPES, MERGE_COLS, MERGE_COL_TYPES, \
                    TRAIN_TRADE_FNM, FULL_TRADE_FNM, DATAPATH
from time    import time

import gc; gc.enable()

class Merge(object):
    def __init__(self, train_fnm, test_fnm):
        self.test_fnm  = test_fnm
        self.train_fnm = train_fnm
        
        self.stats = Stats(train_fnm, test_fnm)
        self.stats_cols = TRADE_COLS
        self.stats_types = TRADE_COL_TYPES
        idx = self.stats.train_df.ticker == ''
        self.stats.train_df = self.stats.train_df.loc[idx]
        
        self.nth_buy      = 1
        self.tns_df_cols  = MERGE_COLS
        self.tns_df_types = MERGE_COL_TYPES
        self.ticker_n_stats_df = empty_dataframe(self.tns_df_cols, 
                                                 self.tns_df_types)

    def extract_n_drop_nth_buy(self):
        test_len_b4 = len(self.stats.test_df)
        
        cols = ['ticker', 'buy_date']
        self.buy_df = self.stats.test_df[cols].groupby(by='ticker').min().reset_index()

        self.stats.test_df = pd.merge(left=self.stats.test_df, 
                                      right=self.buy_df,
                                      on='ticker', how='inner', 
                                      suffixes=('', '_x'))

        idx = self.stats.test_df.buy_date == self.stats.test_df.buy_date_x
        self.buy_df = self.stats.test_df.loc[idx]

        dfs = [self.stats.train_df, self.buy_df]
        self.stats.train_df = pd.concat(dfs)[self.stats_cols]

        self.buy_df = pd.merge(left=self.buy_df, right=self.stats.df,
                               on='ticker', how='inner')[self.tns_df_cols]

        cols=self.stats_cols        
        self.stats.test_df = self.stats.test_df[cols].loc[~idx]

        test_len_after = len(self.stats.test_df)
        log(f'{self.nth_buy:03d}th buy: self.test_df {test_len_b4} ->'
            f' {test_len_after} ({test_len_b4-test_len_after})', True)
        self.nth_buy += 1

        gc.collect()

    def merge_ticker_n_stats(self):

        idx = self.stats.df.ticker == ''
        self.stats.df = self.stats.df.loc[idx]
        self.extract_n_drop_nth_buy()
        self.extract_n_drop_nth_buy()

        while len(self.stats.test_df) > 0:
            test_len = len(self.stats.test_df)
            self.stats.calc_stats()
            self.extract_n_drop_nth_buy()

            dfs = [ self.ticker_n_stats_df, self.buy_df ]
            self.ticker_n_stats_df = pd.concat( dfs )

def merge_main():
    start_time = time()
    log(f'Starting Merger of trades and ticker statistics', True)
    merger = Merge(TRAIN_TRADE_FNM, FULL_TRADE_FNM)
    merger.merge_ticker_n_stats()
    fnm=f'{DATAPATH}trades_n_stats.csv'
    save_dataframe_to_csv(merger.ticker_n_stats_df, fnm)
    log('')
    log('Done.')
    calc_runtime(start_time, True)

if __name__ == "__main__":

    log_filename("merge")

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        merge_main()

 


    