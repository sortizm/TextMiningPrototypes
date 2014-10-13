#!/usr/bin/env python3

"""
Get tweets matching the queries defined in the configuration file

Retrieve and store the tweets, provided by the GET search/tweets endpoint,
which match the queries defined in the configuration file.
SEE: https://dev.twitter.com/rest/reference/get/search/tweets

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
    thm.py [-h] [-l LEVEL] [-o FILE] CONFIGFILE CREDENTIALSFILE

Options:
    -h --help                       show this
    -l --log-level=LEVEL            set the logging level [default: ERROR]
    -o --log-file=FILE              a file to log the output

Loggin levels (from more verbose to less verbose):
    * {0}
    * {1}
    * {2}
    * {3}
    * {4}

Configuration file format:
    topic1 = 'query1', 'query2', ..., 'queryN'
    topic2 = 'query1', 'query2', ..., 'queryN'

    where 'topic' is the the name of the file in which the tweets will be
    stored, and 'query' is a valid Twitter query.
    SEE: https://dev.twitter.com/rest/public/search

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


class TopicMiner(object):
    """
    Retrieve all the tweets of a topic

    get_tweets() return a tuple of Tweet objects
    """

    logger = logging.getLogger('TopicMiner')
    queries = None
    tweets = None
    _t_api = None

    def __init__(self, queries):
        """
        Initialize twitter connection objects

        queries: a set-like object of queries (strings)
        """
        self.logger.info('Creating an API connection...')
        auth = twitter.oauth.OAuth(
                CREDENTIALS['oauth_token'],
                CREDENTIALS['oauth_token_secret'],
                CREDENTIALS['consumer_key'],
                CREDENTIALS['consumer_secret'])
        self._t_api = twitter.Twitter(auth=auth)
        self.logger.info('Twitter object received {0}'.format(self._t_api))
        self.queries = queries

    def _get_tweets(self):
        """
        Set the class attribute tweets to the tuple of Tweet objects, returned
        by the twitter API, that match any of the queries
        """
        self.tweets = list()
        for query in self.queries:
            query_tweets = list()
            self.logger.info('Retrieving statuses with query {0}'.format(query))
            results = self._t_api.search.tweets(q=query, lang='es', count=100)
            query_tweets += [Tweet(status) for status in results['statuses']]
            while True:
                try:
                    next_results = results['search_metadata']['next_results']
                except KeyError:
                    self.logger.info('Retrieved {0} statuses with query {1}'\
                            ''.format(len(query_tweets), query))
                    break  # No more results when next_results does not exist
                # Create a dictionary from next_results:
                # ?max_id=313519052523986943&q=NCAA&include_entities=1
                kwargs = dict([kv.split('=') for kv in next_results[1:].split("&")])
                results = self._t_api.search.tweets(**kwargs)
                query_tweets += [Tweet(status) for status in results['statuses']]
            self.tweets += query_tweets
        self.logger.info('Retrieved a total of {0} statuses'.format(len(self.tweets)))


    def get_tweets(self):
        """Return a tuple of tweets, that match any of the queries"""
        if self.tweets is None:
            self._get_tweets()
        return self.tweets

    def refresh(self):
        """Reload the tweets attribute"""
        self._get_tweets()


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

    if loglevel not in LOG_LEVELS:
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    else:
        logging_level = LOG_LEVELS[loglevel]
        if logfile  is not None:
            logging.basicConfig(filename=logfile, level=logging_level, format=logft)
            with open(logfile, 'a') as lfile:
                lfile.write('\n' + '*'*80 + '\n') # to separate different logs
        else:
            logging.basicConfig(level=logging_level, format=logft)

    logger = logging.getLogger()
    logger.info('Reading credentials file {0}'.format(credentials_file))
    load_credentials(credentials_file)

    logger.info('Reading configuration file {0}'.format(logfile))
    for topicname, queries in load_topics(arguments['CONFIGFILE']):
        logger.debug('Creating TopicMiner object for {0}: queries={1}'\
                ''.format(topicname, queries))
        topicminer = TopicMiner(queries)
        storage = JSONStorage(topicname)
        storage.store(topicminer.get_tweets())


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

def load_topics(configuration_filename):
    """Load the topics and their queries from the configuration file"""
    topic_queries = list()
    with open(configuration_filename, 'r') as configfile:
        for line in configfile.readlines():
            if line.strip()[0] == '#':
                continue  # ignore lines starting with '#'
            assignment = line.split('=')
            if len(assignment) == 2:
                topicname = assignment[0].strip()
                queries = tuple(q.strip() for q in assignment[1].split(','))
                topic_queries.append((topicname, queries))
    return tuple(topic_queries)

if __name__ == '__main__':
    main()
