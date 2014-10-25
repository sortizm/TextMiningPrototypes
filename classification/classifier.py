import nltk
#import numpy
#import matplotlib
from sklearn.svm import SVC
import pstats
import io
import os
import random
from nltk.corpus import brown, stopwords
from collections import Counter
import json

import cProfile as profile


profiler = profile.Profile()
profiler.enable()

################################################################################
datavars = dict()
corpus = brown
CATEGORIES = corpus.categories()
FILENAME = 'cached_data.json'

def get_all_words():
    if 'all_words' not in datavars and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'all_words' not in datavars: # even after loading file
        all_words = sorted(list(
            set(w.lower() for w in corpus.words() if w.isalnum() and len(w) > 3) -
            set(stopwords.words())
            ))
        datavars['all_words'] = all_words
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)
    return datavars['all_words']


def get_samples_category():
    if 'samples_category' not in datavars and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'samples_category' not in datavars: # even after loading file
        samples_category = dict()
        for category in corpus.categories():
            samples = list()
            for fid in corpus.fileids(categories=category):
                count = Counter(corpus.words(fid))
                samples.append(tuple(count[word] for word in datavars['all_words']))
            samples_category[category] = tuple(samples)
        datavars['samples_category'] = samples_category
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)
    return datavars['samples_category']

def get_SVM_samples_predictions(samples_categories):
    if 'svm_samples_predictions' not in datavars and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'svm_samples_predictions' not in datavars: # even after loading file
        svm_samples = list()
        svm_predictions = list()
        for category, samples in samples_categories.items():
            cat = CATEGORIES.index(category)
            for samp in samples:
                svm_samples.append(samp)
                svm_predictions.append(cat)
        indexes = list(range(len(svm_samples)))
        random.shuffle(indexes)
        svm_samples_shuf = list()
        svm_predictions_shuf = list()
        len80pc = int(len(indexes) * 0.8)
        for i in indexes:
            svm_samples_shuf.append(svm_samples[i])
            svm_predictions_shuf.append(svm_predictions[i])
        # 80% of samples will be used for training, and the remaining 20% for testing
        svm_samples_predictions = (tuple(svm_samples_shuf[:len80pc]), tuple(svm_predictions_shuf[:len80pc]),
                tuple(svm_samples_shuf[len80pc+1:]), tuple(svm_predictions_shuf[len80pc+1:]))
        datavars['svm_samples_predictions'] = svm_samples_predictions
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)
    return datavars['svm_samples_predictions']

ALL_WORDS = get_all_words()
SAMPLES_CATEGORY = get_samples_category()

TRAIN_SAMP, TRAIN_PRED, TEST_SAMP, TEST_PRED = get_SVM_samples_predictions(SAMPLES_CATEGORY)
print('Training...')
CLASSIFIER = SVC(kernel='rbf')
CLASSIFIER.fit(TRAIN_SAMP, TRAIN_PRED)
print('Calculating mean accuracy')
print(CLASSIFIER.score(TEST_SAMP, TEST_PRED))

################################################################################

profiler.disable()
stream = io.StringIO()
ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
ps.print_stats()
print(stream.getvalue())
