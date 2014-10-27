from __future__ import print_function
from time import time
import os
import io
import pstats
import nltk
import numpy
import matplotlib
import sklearn
import json
from scipy import sparse
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem import snowball
from sklearn.decomposition import NMF
#from sklearn.cluster import KMeans

import cProfile as profile

profiler = profile.Profile()
profiler.enable()

"""

"""
#C:\Users\Ignacio\Desktop\corpusTweetsEng
#C:\Users\Ignacio\Desktop\corpusTweetsSP
#C:\Users\Ignacio\Downloads\corpusPrueba
#C:\Users\Ignacio\Downloads\corpusCine\corpusTxt


from nltk.corpus.reader.plaintext import PlaintextCorpusReader
corpusdir = r"C:\Users\Ignacio\Downloads\corpusPrueba" #CHANGE THIS (local adress of the corpus)
newcorpus = PlaintextCorpusReader(corpusdir, ".*") 

###############

#Global constants and variables

n_topics=7
n_top_words=10

corpus=brown #selecting the corpus to use
datavars = dict()
FILENAME = 'cached_data.json'

files=corpus.fileids()
lenFiles=len(files)


def preprocess(min_percentage,max_percentage):
    """
    Filters the words in the corpus. We discard numbers, very short words and words appearing in less than the percentage of the first arg of the texs 
    and in more than the second arg percentage of the texts.
    return: a 2-dimension list with the filtered words and the filtered texts
    """
    if ('filtered_words' not in datavars or 'wordsTexts' not in datavars) and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'filtered_words' not in datavars or 'wordsTexts' not in datavars: # even after loading file
        print('No json found in the preprocess')
        wordsTexts=[] #list of counts
        filteredWords=[]
        tuplaAux=()
        for textoId in files:
            wordsRep=tuple(w.lower() for w in corpus.words(textoId) if w.isalpha() and len(w)>3)
            wordsTexts.append(wordsRep)
            tuplaAux=tuplaAux+tuple(set(wordsRep))
        
        noRepWords=tuple(set(tuplaAux))
        count=Counter(tuplaAux)
        
        textosMax=numpy.round(lenFiles*max_percentage/100)
        textosMin=numpy.round(lenFiles*min_percentage/100)

        filteredWords=list(a for a in count if count[a]>textosMin and count[a]<textosMax and a not in stopwords.words('spanish'))
        datavars['filtered_words'] = filteredWords
        datavars['wordsTexts'] = wordsTexts
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)
    return [datavars['filtered_words'],datavars['wordsTexts']]
    

def stemming(palabras,wordsTexts):
    """
    Reduces the amount of filtering words keeping only their stems
    return: a list with the words,the texts filtered and the text frecuency (counting only files nto the number of times in each file) of each stem
    """
    if ('stemmed_words' not in datavars or 'stemmed_texts' not in datavars or 'freq_words' not in datavars) and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'stemmed_words' not in datavars or 'stemmed_texts' not in datavars or 'freq_words' not in datavars: # even after loading file
        print('No json found in the stemming')
        stemmer1 = SnowballStemmer("spanish")
        stemWordText=[[] for i in range(len(wordsTexts))]
        stemPalabras=dict()
        for i in palabras:
            try:
                #stemPalabras.append(stemmer1.stem(i))
                stemPalabras[i]=stemmer1.stem(i)
            except:
                pass 
        i=-1
        freqWords=()
        for text in wordsTexts:            
            i=i+1
            #appendList=[]
            for word in text:
                try:
                    stemWordText[i].append(stemPalabras[word])
                    #aux_word=stemmer1.stem(word)
                    #stemWordText[i].append(aux_word)
                except:
                    pass
            freqWords=freqWords+tuple(set(stemWordText[i]))
        freqWords=Counter(freqWords)        
        #palabras=list(set(stemPalabras))
        palabras=list(set(stemPalabras.values()))
        wordsTexts=stemWordText
        datavars['stemmed_words'] = palabras
        datavars['stemmed_texts'] = wordsTexts
        datavars['freq_words'] = freqWords
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)

    return [datavars['stemmed_words'],datavars['stemmed_texts'],datavars['freq_words']]

def create_matrix(palabras,wordsTexts,freqWords):
    """
    Creates a matrix with the TF*IDF heuristic applied. The matrix rows are the words and the matrix columns are the corpus files.
    return: the matrix and two lists with the labels of the rows and columns
    """
    filas=lenFiles #numero de filas de la matriz
    columnas=len(palabras)
    idFilas=[]
    idColumnas=[]
    matriz=[[0]*columnas for i in range(filas)]
    for i in range(columnas) :
        idColumnas.append(palabras[i])
    for i in range(filas):
        idFilas.append(files[i])

    for i in range(lenFiles):
        fD1=nltk.FreqDist(wordsTexts[i])
        for e in idColumnas:
            auxValor=fD1.get(e)
            if auxValor!=None:
                columnaAux=idColumnas.index(e)
                matriz[i][columnaAux]=numpy.log(auxValor+1) * numpy.log(filas/freqWords[e])
    matriz=numpy.array(matriz)
    return [matriz,idFilas,idColumnas]
  
def factorization_NMF(matriz):
    """
    Applies the non-negative matrix factorization
    return: The characteristics matrix and the weight matrix
    """      
    nmf = NMF(n_components=n_topics, random_state=1).fit(matriz)
    F=nmf.components_ #matriz de caracteristicas (filas:caracteristicas , columnas:palabras)
    W=nmf.fit_transform(matriz) #matriz de pesos (filas:textos , columnas:caracteristicas)
    return [F,W]

  
def print_topics(F,W,idFilas,idColumnas):
    """
    Prints the topics and the text topics in two files
    """  
    with open("Topics.txt", "w") as text_file:
        for topic_idx, topic in enumerate(F):
            print("Topic {}: ".format(topic_idx), file=text_file)
            print(" ".join([idColumnas[i] for i in topic.argsort()[:-n_top_words - 1:-1]]), file=text_file) 
        text_file.close()
    with open("TextTopics.txt", "w") as text_file:
        for text, category in enumerate(W):
            print("Text #{}:".format(idFilas[text]),end=' ',file=text_file)
            print("Category #{}".format(category.argsort()[len(category)-1]),file=text_file) #tomo el mayor(esta el ultimo al ordenarlo)
        text_file.close()

#############
#Delete the jSon file is the corpus is changed or modified
t0 = time()
print("Preprocessing")
[prep_words,prep_texts]=preprocess(5,85)
print("done in %0.3fs." % (time() - t0))
print("Stemming")
stem_words,stem_texts,stem_freq=stemming(prep_words,prep_texts)
print("done in %0.3fs." % (time() - t0))
print("Creating data matrix")
matrix,idRows,idColumns=create_matrix(stem_words,stem_texts,stem_freq)
print("done in %0.3fs." % (time() - t0))
print("Non-negative factorization")
ch_matrix,weight_matrix=factorization_NMF(matrix)
print("done in %0.3fs." % (time() - t0))
print("Printing results")
print_topics(ch_matrix,weight_matrix,idRows,idColumns)
print("done in %0.3fs." % (time() - t0))

profiler.disable()
stream= open('stats.txt','w') 
ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
ps.print_stats()
stream.close()