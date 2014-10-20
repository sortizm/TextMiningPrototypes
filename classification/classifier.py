import nltk
#import numpy
#import matplotlib
#import sklearn
import pstats
import io
import os
import pickle
from nltk.corpus import brown, stopwords
from collections import Counter

import cProfile as profile


profiler = profile.Profile()
profiler.enable()

################################################################################
datavars = dict()
corpus = brown
CATEGORIES = corpus.categories()
FILENAME = 'classifier.dat'

def get_all_words():
    if 'all_words' not in datavars and os.path.exists(FILENAME):
        with open(FILENAME, 'rb') as datafile:
            datavars.update(pickle.load(datafile))
    if 'all_words' not in datavars: # even after loading file
        all_words = sorted(list(
            set(w.lower() for w in corpus.words() if w.isalnum()) -
            set(stopwords.words())
            ))
        datavars['all_words'] = all_words
        with open(FILENAME, 'wb') as datafile:
            pickle.dump(datavars, datafile)
    return datavars['all_words']

def get_samples_category(all_words):
    if 'samples_category' not in datavars and os.path.exists(FILENAME):
        with open(FILENAME, 'rb') as datafile:
            datavars.update(pickle.load(datafile))
    if 'samples_category' not in datavars: # even after loading file
        samples_category = dict()
        for category in corpus.categories():
            samples = list()
            for fid in corpus.fileids(categories=category):
                count = Counter(corpus.words(fid))
                samples.append(tuple(count[word] for word in all_words))
            samples_category[category] = tuple(samples)
        datavars['samples_category'] = samples_category
        with open(FILENAME, 'wb') as datafile:
            pickle.dump(datavars, datafile)
    return datavars['samples_category']

def get_SVM_samples(samples_categories):
    svm_samples = list()
    svm_predictions = list()
    for category,samples in samples_categories.items():
        cat = CATEGORIES.index(category)
        for samp in samples:
            svm_samples.append(samp)
            svm_predictions.append(cat)
    return tuple(svm_samples), tuple(svm_predictions)

SAMPLES_CATEGORY = get_samples_category(get_all_words())

SAMPLES, PREDICTIONS = get_SVM_samples(SAMPLES_CATEGORY)

################################################################################

profiler.disable()
stream = io.StringIO()
ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
ps.print_stats()
print(stream.getvalue())
