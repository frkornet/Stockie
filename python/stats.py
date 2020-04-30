import pandas as pd
import numpy as np
from symbols import TRAIN_TRADE_FNM, TEST_TRADE_FNM, STATS_FNM, LOGPATH, \
                    PICPATH, STAT_COLS,STAT_COL_TYPES, TRADE_COLS, \
                    TRADE_COL_TYPES, STAT_BATCH_SIZE
from util    import open_logfile, log, is_holiday, get_current_day_and_time, \
                    empty_dataframe, dict_to_dataframe, save_dataframe_to_csv,\
                    add_months

from tqdm    import tqdm

import matplotlib.pyplot as plt

import gc; gc.enable()
import warnings; warnings.filterwarnings("ignore")

class Stats(object):

    def __read_trades__(self, fnm):
        possible_trades_df = pd.read_csv(fnm)
        return possible_trades_df

    def augment_train_df(self):
        self.train_df['gain'] = self.train_df.sell_close \
                              - self.train_df.buy_close
        self.train_df['pct_gain'] = (self.train_df.gain \
                                  / self.train_df.buy_close) * 100
        self.train_df['ret'] = 1 + self.train_df['pct_gain']/100
        self.train_df['dret'] = round(self.train_df.ret \
                             ** (1/self.train_df.trading_days) - 1, 4) \
                              * 100

        self.gain_df = self.train_df.loc[self.train_df.gain >  0]
        self.desired_df = self.train_df.loc[self.train_df.dret >  0.5]
        self.loss_df = self.train_df.loc[self.train_df.gain <  0]
        self.zero_df = self.train_df.loc[self.train_df.gain == 0]

        self.total_tickers = set(self.train_df.ticker.unique())
        self.desired_tickers = set(self.desired_df.ticker.unique())
        self.gain_tickers  = set(self.gain_df.ticker.unique())
        self.loss_tickers  = set(self.loss_df.ticker.unique())
        self.zero_tickers  = set(self.zero_df.ticker.unique())

    def reset_trade_files(self, train_fnm, test_fnm):
        self.train_fnm = train_fnm
        self.train_df = self.__read_trades__(train_fnm)
        assert len(self.train_df) > 0, "training data set cannot be empty"
        gc.collect()

        self.test_fnm  = test_fnm
        self.test_df = self.__read_trades__(test_fnm)
        gc.collect()

    def parse_keep(self, keep):
        months = int(keep[0:-1])
        assert months >= 0, "only positive amounts are allowed!"
        if keep[-1] == "y" or keep[-1]== "Y":
            months = months * 12
        else:
            assert keep[-1] == "m" or keep[-1]== "M", "only m and y allowed!"
        self.months = months

    def __init__(self, train_fnm, test_fnm, keep="0m"):
        self.reset_trade_files(train_fnm, test_fnm)
        self.ticker = ''
        self.batched_days = []
        self.batched_idx  = self.test_df.sell_date == ''
        self.total_tickers = set(self.train_df.ticker.unique())
        self.parse_keep(keep)

    def desired_stats(self):
        idx = self.train_df.dret > 0.5
        ddf = self.train_df.loc[idx].groupby(by='ticker')['dret'].count().reset_index()
        ddf.columns = [ 'ticker', 'cnt_desired']
        
        # create zeros for missing tickers -> edf
        tickers_to_add = list(self.total_tickers - self.desired_tickers)
        tlen = len(tickers_to_add)
        zero_list_int   = [int(0)] * tlen

        t_dict = {'ticker'      : tickers_to_add, 
                  'cnt_desired' : zero_list_int
                 }
        edf = pd.DataFrame(t_dict)

        #concatenate gdf and edf -> gdf
        self.ddf = pd.concat([ddf, edf])

    def gain_stats(self):
        # group by ticker and calculate stats -> tdf
        aggs=['count', 'min', 'max', 'std', 'mean', 'sum']
        tdf = self.gain_df.groupby(by='ticker').agg(aggs)[['gain_pct', 'trading_days']].reset_index()
        tdf.columns=['ticker', 'cnt_gain', 'min_pct_gain', 'max_pct_gain', 'std_pct_gain', 
                     'mean_pct_gain', 'sum_pct_gain',
                     'cnt_trading_days', 'min_trading_days', 'max_trading_days', 'std_trading_days',
                     'mean_day_gain', 'sum_day_gain']

        # set null for std_pct_gain to zero in case there is only 1 row for ticker
        idx = tdf.std_pct_gain.isna() == True
        tdf.std_pct_gain.loc[idx] = 0

        # calculate return per ticker -> rdf
        rdf = self.gain_df.groupby(by='ticker')['ret'].prod().reset_index()
        rdf.columns=['ticker', 'ret']

        # inner join join gdf and rdf -> gdf
        gdf = pd.merge(left=tdf, right=rdf, on='ticker', how='inner')
        gdf['gain_daily_ret'] = round((gdf.ret ** (1/gdf.sum_day_gain) - 1) * 100, 2)

        # only keep needed columns
        cols_to_keep = ['ticker', 'cnt_gain', 'min_pct_gain', 'max_pct_gain', 'std_pct_gain', 'mean_pct_gain',
                        'mean_day_gain', 'gain_daily_ret']
        gdf = gdf[cols_to_keep]

        # create zeros for missing tickers -> edf
        tickers_to_add = list(self.total_tickers - self.gain_tickers)
        tlen = len(tickers_to_add)
        zero_list_float = [0.0] * tlen
        zero_list_int   = [int(0)] * tlen

        t_dict = {'ticker'         : tickers_to_add, 
                  'cnt_gain'       : zero_list_int,
                  'min_pct_gain'   : zero_list_float,
                  'max_pct_gain'   : zero_list_float,
                  'std_pct_gain'   : zero_list_float,
                  'mean_pct_gain'  : zero_list_float,
                  'mean_day_gain'  : zero_list_float,
                  'gain_daily_ret' : zero_list_float
                 }
        edf = pd.DataFrame(t_dict)

        #concatenate gdf and edf -> gdf
        gdf = pd.concat([gdf, edf])
        self.desired_stats()
        self.gdf = pd.merge(left=gdf, right=self.ddf, on='ticker', how='inner')

    def loss_stats(self):
        # group by ticker and calculate stats -> tdf
        aggs=['count', 'min', 'max', 'std', 'mean', 'sum']
        tdf = self.loss_df.groupby(by='ticker').agg(aggs)[['gain_pct', 'trading_days']].reset_index()
        tdf.columns=['ticker', 'cnt_loss', 'min_pct_loss', 'max_pct_loss', 'std_pct_loss', 'mean_pct_loss',
                     'sum_pct_loss',
                     'cnt_trading_days', 'min_trading_days', 'max_trading_days', 'std_trading_days',
                     'mean_day_loss', 'sum_day_loss']

        # set null for std_pct_gain to zero in case there is only 1 row for ticker
        idx = tdf.std_pct_loss.isna() == True
        tdf.std_pct_loss.loc[idx] = 0

        # calculate return per ticker -> rdf
        rdf = self.loss_df.groupby(by='ticker')['ret'].prod().reset_index()
        rdf.columns=['ticker', 'ret']

        # inner join join gdf and rdf -> gdf
        ldf = pd.merge(left=tdf, right=rdf, on='ticker', how='inner')
        ldf['loss_daily_ret'] = round((ldf.ret ** (1/ldf.sum_day_loss) - 1) * 100, 2)

        # only keep needed columns
        cols_to_keep = ['ticker', 'cnt_loss', 'min_pct_loss', 'max_pct_loss', 'std_pct_loss', 'mean_pct_loss',
                        'mean_day_loss', 'loss_daily_ret']
        ldf = ldf[cols_to_keep]

        # create zeros for missing tickers -> edf
        tickers_to_add = list(self.total_tickers - self.loss_tickers)
        tlen = len(tickers_to_add)
        zero_list_float = [0.0] * tlen
        zero_list_int   = [int(0)] * tlen

        t_dict = {'ticker'         : tickers_to_add, 
                  'cnt_loss'       : zero_list_int,
                  'min_pct_loss'   : zero_list_float,
                  'max_pct_loss'   : zero_list_float,
                  'std_pct_loss'   : zero_list_float,
                  'mean_pct_loss'  : zero_list_float,
                  'mean_day_loss'  : zero_list_float,
                  'loss_daily_ret' : zero_list_float
                 }
        edf = pd.DataFrame(t_dict)

        #concatenate gdf and edf -> gdf
        self.ldf = pd.concat([ldf, edf], sort=False)

    def zero_stats(self):
        # group by ticker and calculate stats -> tdf
        aggs=['count', 'mean']
        tdf = self.zero_df.groupby(by='ticker').agg(aggs)[['trading_days']].reset_index()
        tdf.columns=['ticker', 'cnt_zero', 'mean_day_zero']

        # create zeros for missing tickers -> edf
        tickers_to_add = list(self.total_tickers - self.zero_tickers)
        tlen = len(tickers_to_add)
        zero_list_float = [0.0] * tlen
        zero_list_int   = [int(0)] * tlen

        t_dict = {'ticker'         : tickers_to_add, 
                  'cnt_zero'       : zero_list_int,
                  'mean_day_zero'  : zero_list_float
                 }
        edf = pd.DataFrame(t_dict)

        #concatenate gdf and edf -> gdf
        self.zdf = pd.concat([tdf, edf], sort=False)

    def total_stats(self):
        # group by ticker and calculate stats -> tdf
        aggs=['count', 'sum', 'mean']
        tdf = self.train_df.groupby(by='ticker').agg(aggs)[['trading_days']].reset_index()
        tdf.columns=['ticker', 'total_cnt', 'total_days', 'mean_day']

        # calculate return per ticker -> rdf
        rdf = self.train_df.groupby(by='ticker')['ret'].prod().reset_index()
        rdf.columns=['ticker', 'ret']

        # inner join join gdf and rdf -> gdf
        tdf = pd.merge(left=tdf, right=rdf, on='ticker', how='inner')
        tdf['daily_ret'] = round((tdf.ret ** (1/tdf.total_days) - 1) * 100, 2)

        # only keep needed columns
        cols_to_keep = ['ticker', 'total_cnt', 'total_days', 'mean_day', 'daily_ret']
        self.tdf = tdf[cols_to_keep]
        
    def create_stats(self):
        self.augment_train_df()

        self.gain_stats()
        log(f"ddf: {len(self.ddf)}")
        log(f"gdf: {len(self.gdf)}")
        self.loss_stats()
        log(f"ldf: {len(self.ldf)}")
        self.zero_stats()
        log(f"zdf: {len(self.zdf)}")
        self.total_stats()
        log(f"tdf: {len(self.tdf)}")
        
        cdf = pd.merge(left=self.gdf, right=self.ldf, on='ticker', how='inner' )
        cdf = pd.merge(left=cdf, right=self.zdf, on='ticker', how='inner' )
        cdf = pd.merge(left=cdf, right=self.tdf, on='ticker', how='inner' )
        cdf['pct_desired'] = round((cdf.cnt_desired / cdf.total_cnt) * 100, 2)
        cdf['gain_ratio'] = round((cdf.cnt_gain / cdf.total_cnt) * 100, 2)
        cdf['good'] = 0

        self.df = cdf.copy()

        gc.collect()
 
    def calc_stats(self):
        if self.months > 0:
            max_date = max(self.train_df.sell_date)
            latest_date = add_months(max_date, -self.months)
            idx = self.train_df.sell_date <= latest_date
            self.train_df = self.train_df.loc[~idx]

        self.create_stats()

    def __sum_n_count__(self, df):
        if len(df) == 0:
            return 0, 0

        aggs = ['sum', 'count']
        s, c = df['gain'].agg(aggs)
        return s, c

    def __calc_n_print_ratios__(self, df):
        gain_s, gain_c = self.__sum_n_count__(df.loc[df.gain > 0])
        loss_s, loss_c = self.__sum_n_count__(df.loc[df.gain < 0])
        zero_s, zero_c = self.__sum_n_count__(df.loc[df.gain == 0])

        gain_s = round(gain_s, 2)
        loss_s = round(loss_s, 2)

        log('')
        log(f'gain_sum={gain_s} gain_cnt={gain_c}')
        log(f'loss_sum={loss_s} loss_cnt={loss_c}')
        log(f'zero_sum={zero_s} zero_cnt={zero_c}')
        log('')
        gain_share = (gain_c / (gain_c + loss_c + zero_c)) * 100
        gain_share = round(gain_share,2)
        log(f'gain_share={gain_share} %')
        log('')

    def train_df_ratios(self):
        log('ratios for all tickers self.train_df:')
        self.__calc_n_print_ratios__(self.train_df)

    def __log_heuristics_stats__(self):
        log(f'len(self.df)={len(self.df)}')
        log(f'len(self.df[idx])={len(self.df[self.idx])}')
        ratio = round(len(self.df[self.idx])/len(self.df), 2)
        log(f'ratio={ratio}')

    def heuristic(self, threshold, verbose=True):
        self.threshold = threshold
        self.idx = (self.df.daily_ret  > 0.0) \
                 & (self.df.pct_desired > threshold)
        
        self.df.good = 0
        self.df.loc[self.idx, 'good'] = 1

        if verbose == True:
            self.__log_heuristics_stats__()

    def __good_subset_posible_trades__(self):

        cols = ['ticker', 'good']
        tdf = pd.merge(left=self.train_df, right=self.df[cols], \
                       on='ticker', how='inner')
        self.good_df = tdf.loc[tdf.good == 1]

    def __log_save_stats__(self):

        self.__good_subset_posible_trades__()
        good_len = len(self.good_df)
        train_df_len = len(self.train_df)

        df_len = len(self.df)
        df_good_sum = sum(self.df.good)

        share= round(good_len / train_df_len * 100, 0)

        log('')
        log(f'total ticker count={df_len}')
        log(f'good tickers count={df_good_sum}')
        log('')
        log(f'all ticker trades ={train_df_len}')
        log(f'good ticker trades={good_len}')
        log(f'good ticker share ={share}%')
        log('')

    def save_good_tickers(self, fnm, verbose=True):

        #self.__set_good_tickers__()
        if verbose == True:
            self.__log_save_stats__()
        save_dataframe_to_csv(self.df, fnm)

    def __drop_and_calc_stats__(self):
        # drop done by calc_stats
        self.calc_stats()

        self.heuristic(self.threshold)
        self.batched_days = []
        self.batched_idx  = self.test_df.sell_date == ''

    def add_day(self, trading_date):

        idx = self.test_df.sell_date == trading_date
        test_len = len(self.test_df.loc[idx])
        if test_len == 0:
            return
        
        self.batched_days.append(trading_date)
        self.batched_idx |= self.test_df.sell_date == trading_date
        train_tickers = set(self.test_df[self.batched_idx].ticker.unique())
        good_tickers = set(self.df.loc[self.df.good == 1].ticker.unique())
        common_tickers = train_tickers & good_tickers
        if len(common_tickers) < len(good_tickers) * STAT_BATCH_SIZE:
            return

        tickers_to_process = list(self.test_df \
            .loc[self.batched_idx].ticker.unique())

        log(f'Processing {self.batched_days}')
        log(f'Unique tickers={len(tickers_to_process)}')

        self.train_df = pd.concat([self.train_df, 
                                   self.test_df.loc[self.batched_idx]])
        
        self.__drop_and_calc_stats__()

def test_add_day(stats):
    log('Test add_day() functionality for non-existing trading date:')
    stats.add_day('2100-04-28')

    dates_to_process = sorted(list(stats.test_df.sell_date.unique()))

    for d in tqdm(dates_to_process, "Dates:"):
        stats.add_day(d)

def stats_main():
    log("Starting stats.py")
    log('')
    stats = Stats(TRAIN_TRADE_FNM, TEST_TRADE_FNM, "24m")

    log('Calculating individual ticker stats...\n')
    stats.calc_stats()

    log('Setting threshold for pct_desired > 10...')
    stats.heuristic(10)

    log('Saving the ticker stats (including good)...')
    stats.save_good_tickers(STATS_FNM)

    test_add_day(stats)

if __name__ == "__main__":

    log_fnm = "stats"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        stats_main()
