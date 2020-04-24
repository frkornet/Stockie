import pandas                as pd
import numpy                 as np
import gc; gc.collect()

from   tqdm                  import tqdm
from   util                  import open_logfile, log, is_holiday, \
                                    get_current_day_and_time
from   trade                 import ticker_trades
from symbols                 import LOGPATH, DATAPATH, STOCKS_FNM, QA_PERIOD, \
                                    YF_TIMES, SLEEP_TIME, QA_SET

def qa_stocks():
    """
    A single run through all the stocks. It checks that the stock Close price
    can be smoothed. If not, it adds it to an exclude list that is returned to 
    the caller at the end.
    """
    stocks = pd.read_csv(STOCKS_FNM)
    exclude_set = set()

    tickers = sorted(list(stocks.TICKER))
    for ticker in tqdm(tickers):
        success, _, _ = ticker_trades(ticker, False)
        gc.collect()

        if success == False:
           log(f'- adding ticker {ticker} to exclude set')
           exclude_set.add(ticker)

    return exclude_set

def qa_single_run(run):
    """
    Single run of qa_run_set(). Returns exclude set to caller.
    """
    log('')
    log(f'Run: {run+1}')
    log('')
    exclude_set = qa_stocks()
    log('')
    log(f'{exclude_set}')
    log(f'{len(exclude_set)}')
    log('')
    return exclude_set  

def qa_run_set(times):
    """
    Run through qa_stocks `times` and build an overall exclude set
    that consists of a union of all runs.
    """
    complete_set = set()
    for i in range(times):
        exclude_set = qa_single_run(i)
        complete_set = complete_set | exclude_set

    return complete_set   


def qa_main():
    complete_exclude_set = qa_run_set(QA_SET)
    log('', True)
    log('Complete exclude set:', True)
    exclude_list = sorted(list(complete_exclude_set))
    log(f'{exclude_list}', True)
    log(f'len(exclude_list)={len(exclude_list)}', True)

    exclude_fnm = f'{DATAPATH}/exclude.csv'
    log('Saving exclude list to {exclude_fnm}')
    ticker_dict = {'ticker' : sorted(list(complete_exclude_set))}
    log('')
    log(f'ticker_dict={ticker_dict}')
    log('')
    exclude_df = pd.DataFrame( ticker_dict )
    log(f'ticker_dict={exclude_df}')
    log(f'len(ticker_df)={len(exclude_df)}')
    exclude_df.to_csv(exclude_fnm, index=False)

    log('')
    log('Done')

if __name__ == "__main__":

    log_fnm = "qa"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        main()

 