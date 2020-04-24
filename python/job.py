from qa         import qa_main
from trade      import trade_main
from stats      import stats_main
from backtest   import backtest_main
from util       import open_logfile, log, get_current_day_and_time, is_holiday
from symbols    import LOGPATH

def job_main():
    log('Starting main Stockie job.')

    log('Starting qa_main().')
    qa_main()
    log('qa_main() is done.')

    log('Starting trade_main().')
    trade_main()
    log('trade_main() is done.')

    log('Starting stats_main().')
    stats_main()
    log('stats_main() is done.')

    log('Starting bactest_main().')
    backtest_main()
    log('backtest_main() is done.')

    log('')
    log('Done.')

if __name__ == "__main__":

    log_fnm = "job"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        job_main()