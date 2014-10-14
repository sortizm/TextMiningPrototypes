#!/usr/bin/env python3

"""
Get tweets using the Twitter streaming API

Author: Steven Ortiz
"""


import twitter
import docopt
import logging
import sys

from os.path import isfile

try:
    import simplejson as json
except ImportError:
    import json

CREDENTIALS = dict()

LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
        }

USAGE = """
Usage:
    tsm.py [-h] [-m LIMIT] [-l LEVEL] [-o FILE] CREDENTIALSFILE

Options:
    -h --help                       show this
    -l --log-level=LEVEL            set the logging level [default: ERROR]
    -o --log-file=FILE              a file to log the output
    -m --max=LIMIT                  maximum number of tweets to download

Loggin levels (from more verbose to less verbose):
    * {0}
    * {1}
    * {2}
    * {3}
    * {4}
""".format(*LOG_LEVELS.keys())


class JSONStorage(object):
    """
    JSON storage class

    store(tweets): store the tweets to the file specified on creation in JSON
                   format
    """

    logger = logging.getLogger('JSONStorage')
    filename = None

    def __init__(self, filename):
        """
        Initialize file storage object

        filename: the name of the file that will be used as a storage unit
        """
        self.filename = filename + '.json'

    def store(self, tweets):
        """
        Store the tweets in JSON format in the file specified at creation

        tweets: the tweets that will be stored
        """
        prev_tweets = dict()
        if isfile(self.filename):
            with open(self.filename, 'r') as storefile:
                self.logger.info('Loading tweets from previous session...')
                prev_tweets = json.load(storefile)
        self.logger.info('Opening {0} to write {1} tweets'.format(
            self.filename, len(tweets)))
        with open(self.filename, 'w') as storefile:
            json.dump(self._tweets_to_dict(tweets, prev_tweets), storefile, indent=3)

    def _tweets_to_dict(self, tweets, prev_tweets_dict):
        """
        Convert a set-like object of Tweet objects to a dictionary representation
        """
        merged_tweets_dict = prev_tweets_dict.copy()
        tweets_dict = dict()
        for twt in tweets:
            twtdict = {'text': twt.text}
            tweets_dict[twt.identifier] = twtdict
        merged_tweets_dict.update(tweets_dict)
        return merged_tweets_dict


class Tweet(object):
    """
    Class that represents a tweet
    """

    logger = logging.getLogger('Tweet')
    identifier = None
    text = None

    def __init__(self, status):
        """
        Create the Tweet object

        status: a Twitter API status
        """
        self.identifier = str(status['id'])
        self.text = status['text']


class StreamMiner(object):
    """
    Retrieve tweets from the Twitter streaming API until the limit is reached,
    an interruption from keyboard is received (^C), or the stream connection is
    closed.

    get_tweets() return a tuple of Tweet objects
    """

    logger = logging.getLogger('StreamMiner')
    limit = None
    tweets = None
    _t_api = None

    def __init__(self, limit):
        """
        Initialize twitter connection objects

        limit = maximum number of tweets to download
        """
        self.logger.info('Creating an API connection...')
        auth = twitter.oauth.OAuth(
                CREDENTIALS['oauth_token'],
                CREDENTIALS['oauth_token_secret'],
                CREDENTIALS['consumer_key'],
                CREDENTIALS['consumer_secret'])
        self._t_api = twitter.TwitterStream(auth=auth)
        self.logger.debug('Twitter object received {0}'.format(self._t_api))
        self.limit = 0 if limit < 0 else limit

    def _get_tweets(self):
        """
        Set the class attribute tweets to the list of tweets retrieved from the
        Twitter stream
        """
        all_locations = '-180,-90,180,90'
        self.tweets = list()
        self.logger.info('Retrieving statuses from Twitter stream API')
        count = 0
        try:
            for status in self._t_api.statuses.filter(locations=all_locations, language='es'):
                self.logger.debug('Status `{0}` received \n{1}'.format(count+1, status))
                if 'text' in status and count != self.limit:
                    self.tweets.append(Tweet(status))
                else:
                    self.logger.info('Retrieved a total of {0} statuses'.format(count))
                    return
                count += 1
        except KeyboardInterrupt:
            self.logger.info('Retrieved a total of {0} statuses'.format(count))

    def get_tweets(self):
        """Return the list of tweets retrieved from the Twitter stream"""
        if self.tweets is None:
            self._get_tweets()
        return self.tweets


def main():
    """
    Parse arguments, read config file, and initialize logger object,
    and miner objects
    """
    logft = '%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s -- %(message)s'
    arguments = docopt.docopt(USAGE)
    loglevel = arguments['--log-level']
    logfile = arguments['--log-file']
    credentials_file = arguments['CREDENTIALSFILE']
    try:
        limit = int(arguments.get('--max', 0))
    except ValueError:
        print("limit must be a number\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    if loglevel not in LOG_LEVELS:
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    else:
        logging_level = LOG_LEVELS[loglevel]
        if logfile is not None:
            logging.basicConfig(filename=logfile, level=logging_level, format=logft)
            with open(logfile, 'a') as lfile:
                lfile.write('\n' + '*'*80 + '\n') # to separate different logs
        else:
            logging.basicConfig(level=logging_level, format=logft)

    logger = logging.getLogger()
    logger.info('Reading credentials file {0}'.format(credentials_file))
    load_credentials(credentials_file)

    streamminer = StreamMiner(limit)
    storage = JSONStorage('tweets')
    storage.store(streamminer.get_tweets())


def load_credentials(credentials_filename):
    """
    Load credentials from credentials_file
    one credential per line, keys and values are space separated

    Go to http://dev.twitter.com/apps/new to create an app and get values
    for these credentials.

    See https://dev.twitter.com/docs/auth/oauth for more information
    on Twitter's OAuth implementation.
    """
    with open(credentials_filename, 'r') as cfile:
        for credential_line in cfile:
            key, val = credential_line.strip().split(' ')
            if key.lower() == 'consumer_key':
                CREDENTIALS['consumer_key'] = val
            elif key.lower() == 'consumer_secret':
                CREDENTIALS['consumer_secret'] = val
            elif key.lower() == 'oauth_token':
                CREDENTIALS['oauth_token'] = val
            elif key.lower() == 'oauth_token_secret':
                CREDENTIALS['oauth_token_secret'] = val

if __name__ == '__main__':
    main()
