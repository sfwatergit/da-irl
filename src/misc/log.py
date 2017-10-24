import datetime as dt
import logging as lg
import os
import sys
import unicodedata

# write log to file and/or to console
_log_file = False
_log_console = False
_log_level = lg.INFO
_log_name = 'deepmarl'
_log_filename = 'deepmarl_log'
_logs_folder = 'logs'


def log(message, level=None, name=None, filename=None):
    """
    Write a message to the log file and/or print to the the console.

    Parameters
    ----------
    message : string, the content of the message to log
    level : int, one of the logger.level constants
    name : string, name of the logger
    filename : string, name of the log file

    Returns
    -------
    None
    """

    if level is None:
        level = _log_level
    if name is None:
        name = _log_name
    if filename is None:
        filename = _log_filename

    # if logging to file is turned on
    if _log_file:
        # get the current logger (or create a new one, if none), then log message at requested level
        logger = get_logger(level=level, name=name, filename=filename)
        if level == lg.DEBUG:
            logger.debug(message)
        elif level == lg.INFO:
            logger.info(message)
        elif level == lg.WARNING:
            logger.warning(message)
        elif level == lg.ERROR:
            logger.error(message)

    # if logging to console is turned on, convert message to ascii and print to the console
    if _log_console:
        # capture current stdout, then switch it to the console, print the message, then switch back to what had been the stdout
        # this prevents logging to notebook - instead, it goes to console
        standard_out = sys.stdout
        sys.stdout = sys.__stdout__

        # convert message to ascii for console display so it doesn't break windows terminals
        message = unicodedata.normalize('NFKD', make_str(message)).encode('ascii', errors='replace').decode()
        print(message)
        sys.stdout = standard_out


def get_logger(level=None, name=None, filename=None):
    """
    Create a logger or return the current one if already instantiated.

    Parameters
    ----------
    level : int, one of the logger.level constants
    name : string, name of the logger
    filename : string, name of the log file

    Returns
    -------
    logger : logger.logger
    """

    if level is None:
        level = _log_level
    if name is None:
        name = _log_name
    if filename is None:
        filename = _log_filename

    logger = lg.getLogger(name)

    # if a logger with this name is not already set up
    if not getattr(logger, 'handler_set', None):

        # get today's date and construct a log filename
        todays_date = dt.datetime.today().strftime('%Y_%m_%d')
        log_filename = '{}/{}_{}.log'.format(_logs_folder, filename, todays_date)

        # if the logs folder does not already exist, create it
        if not os.path.exists(_logs_folder):
            os.makedirs(_logs_folder)

        # create file handler and log formatter and set them up
        handler = lg.FileHandler(log_filename, encoding='utf-8')
        formatter = lg.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.handler_set = True

    return logger


def make_str(value):
    """
    Convert a passed-in value to unicode if Python 2, or string if Python 3.

    Parameters
    ----------
    value : any type, the value to convert to unicode/string

    Returns
    -------
    value : unicode or string
    """
    try:
        # for python 2.x compatibility, use unicode
        return unicode(value)
    except:
        # python 3.x has no unicode type, so if error, use str type
        return str(value)
