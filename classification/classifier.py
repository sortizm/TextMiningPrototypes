import nltk
#import numpy
#import matplotlib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pstats
import io
import os
import re
import random
from nltk.corpus import brown, stopwords
from collections import Counter
import json

from nltk.stem import snowball

import cProfile as profile


profiler = profile.Profile()
profiler.enable()

################################################################################
datavars = dict()
corpus = brown
FID_DIRECTORY = '/home/steven/nltk_data/corpora/brown'
CATEGORIES = corpus.categories()
FILENAME = 'cached_data.json'

def get_all_words():
    if 'all_words' not in datavars and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'all_words' not in datavars: # even after loading file
        stemmer = snowball.EnglishStemmer(ignore_stopwords=True)
        all_words = sorted(list(
            set(stemmer.stem(w) for w in corpus.words() if w.isalnum() and len(w) > 3) -
            set(stopwords.words())
            ))
        datavars['all_words'] = all_words
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)
    return datavars['all_words']

def shuffle_list(lst, seed=None):
    random.seed(a=seed)
    random.shuffle(lst)

def split_list(lst, percent=0.5, part=None):
    end_index_first = int(len(lst) * percent)
    part1, part2 = lst[:end_index_first], lst[end_index_first+1:]
    if part == 1:
        return part1
    elif part == 2:
        return part2
    return part1, part2

def get_samples_predictions(all_words, percent):

    def tokenizer(string):
        stemmer = snowball.EnglishStemmer(ignore_stopwords=True)
        regex = re.compile('\w\w+')
        return tuple(stemmer.stem(w) for w in regex.findall(string))

    vectorizer = TfidfVectorizer(
            input='filename',
            tokenizer=tokenizer,
            ngram_range=(1, 3),
            stop_words=stopwords.words(),
            max_df=0.95, # ignore words with a term frequency higher than 95% (corpus specific stopwords)
            vocabulary=all_words,
            use_idf=True, # use inverse-document-frequency reweighting
            sublinear_tf=True # tf is 1-log(tf)
            )
    sample_fids, predictions = list(), list()
    for category in CATEGORIES:
        for fid in corpus.fileids(categories=category):
            sample_fids.append(os.path.join(FID_DIRECTORY, fid))
            predictions.append(CATEGORIES.index(category))
    shuffle_list(sample_fids, seed=123)
    shuffle_list(predictions, seed=123)
    training_fids, test_fids = split_list(sample_fids, percent=percent)
    training_samples = vectorizer.fit_transform(training_fids)
    test_samples = vectorizer.fit_transform(test_fids)
    training_predictions, test_predictions = split_list(predictions, percent=percent)
    return training_samples, training_predictions, test_samples, test_predictions

ALL_WORDS = get_all_words()
TRAINING_SAMPLES, TRAINING_PREDICTIONS, TEST_SAMPLES, TEST_PREDICTIONS = get_samples_predictions(ALL_WORDS, 0.8)

print('Training...')
CLASSIFIER = SVC(kernel='rbf')
CLASSIFIER.fit(TRAINING_SAMPLES, TRAINING_PREDICTIONS)
print('Calculating mean accuracy')
print(CLASSIFIER.score(TEST_SAMPLES, TEST_PREDICTIONS))

################################################################################

profiler.disable()
stream = io.StringIO()
ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
ps.print_stats()
print(stream.getvalue())
