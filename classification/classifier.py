import nltk
#import numpy
#import matplotlib
#import sklearn
import pstats
import io
import os
from nltk.corpus import brown, stopwords
from collections import Counter
from multiprocessing import Pool # pylint: disable=no-name-in-module
from itertools import zip_longest
import json

import cProfile as profile


profiler = profile.Profile()
profiler.enable()

################################################################################
datavars = dict()
corpus = brown
CATEGORIES = corpus.categories()
FILENAME = 'cached_data.json'
CPU_COUNT = os.cpu_count() if os.cpu_count() is not None else 1
FID_CATEGORY = dict()
for category in corpus.categories():
    FIDS = corpus.fileids(categories=category)
    FID_CATEGORY.update(dict(zip_longest(FIDS, [], fillvalue=category)))

def split_list(initial_list, parts):
    # assumes: there are more elements than parts
    split = list()
    for p in range(parts):
        split.append(tuple(initial_list[p::parts]))
    return split

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


def get_samples_from_fileids(fid_list):
    samp = dict()
    for fid in fid_list:
        count = Counter(corpus.words(fid))
        samp.update({FID_CATEGORY[fid]: tuple(count[word] for word in ALL_WORDS)})
    return samp

def get_samples_category():
    if 'samples_category' not in datavars and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'samples_category' not in datavars: # even after loading file
        cpu_jobs = split_list(corpus.fileids(), CPU_COUNT)
        cpu_pool = Pool(CPU_COUNT)
        scd = cpu_pool.map(get_samples_from_fileids, cpu_jobs)
        samples_category = {cat:[d.get(cat) for d in scd] for cat in {cat for d in scd for cat in d}}
        datavars['samples_category'] = samples_category
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)
    return datavars['samples_category']

def get_SVM_samples(samples_categories):
    svm_samples = list()
    svm_predictions = list()
    for category, samples in samples_categories.items():
        cat = CATEGORIES.index(category)
        for samp in samples:
            svm_samples.append(samp)
            svm_predictions.append(cat)
    return tuple(svm_samples), tuple(svm_predictions)

ALL_WORDS = get_all_words()
SAMPLES_CATEGORY = get_samples_category()

SAMPLES, PREDICTIONS = get_SVM_samples(SAMPLES_CATEGORY)

################################################################################

profiler.disable()
stream = io.StringIO()
ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
ps.print_stats()
print(stream.getvalue())
