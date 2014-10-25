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
Importamos el corpus con el que vamos a trabajar
"""
#C:\Users\Ignacio\Desktop\corpusTweets
#C:\Users\Ignacio\Downloads\corpusPrueba


from nltk.corpus.reader.plaintext import PlaintextCorpusReader
corpusdir = r"C:\Users\Ignacio\Downloads\corpusPrueba" #CHANGE THIS (local adress of the corpus)
newcorpus = PlaintextCorpusReader(corpusdir, ".*") 

n_topics=7
n_top_words=10

corpus=newcorpus #selecting the corpus to use
datavars = dict()
FILENAME = 'cached_data.json'
t0 = time() #initializing the timer

files=corpus.fileids()
lenFiles=len(files)
"""
We are going to create a filter to reduce the number of words we are going to be looking at
filtro2: Keeps the words which appear in the range of a percentage (for example: if a word
        appears in less than 5% of the texts or in more than 95% of the text, it is discarded
        because is not relevant enough) 

We can put a ceiling in the frecuency of a word in a text (divide its frecuency to the total number of words maybe) FEATURE REDUCTION??
Elias takes only the words with a certain percentage of appearance in the text (good idea)
Correct the minimum percentage(maybe it s not correct to include it) and reduce the maximun percentage
TRY OTHER ALGORTITHMS

porcentajes que dependan del numero de textos y del numero de categorias (PROBAR: textos*0.5*categorias/250)
"""

#it returns a list with the words filtered and another list with all the words and its repetition in each text

def preprocess():
    if ('filtered_words' not in datavars or 'wordsTexts' not in datavars) and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'filtered_words' not in datavars or 'wordsTexts' not in datavars: # even after loading file
        print('No json found')
        wordsTexts=[] #list of counts
        filteredWords=[]
        tuplaAux=()
        for textoId in files:
            wordsRep=tuple(w.lower() for w in corpus.words(textoId) if w.isalpha() and len(w)>3)
            wordsTexts.append(wordsRep)
            tuplaAux=tuplaAux+tuple(set(wordsRep))
        
        #count=Counter(tuplaAux)
        noRepWords=tuple(set(tuplaAux))
        count=Counter(tuplaAux)

        textosMax=numpy.round(lenFiles*95/100)
        textosMin=numpy.round(lenFiles*5/100)
        #filteredWords=list( set(a for a,b in dicNum.items() if b>=textosMin and b<=textosMax)-set(stopwords.words('english')) )
        #f= set(a for a,b in dicNum.items() if b>=textosMin and b<=textosMax)
        #filteredWords=list(f)
        filteredWords=list(a for a in count if count[a]>=textosMin and count[a]<=textosMax and a not in stopwords.words('spanish'))
        datavars['filtered_words'] = filteredWords
        datavars['wordsTexts'] = wordsTexts
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)
    return [datavars['filtered_words'],datavars['wordsTexts']]
    


def stemming(palabras,wordsTexts):
    stemmer1 = SnowballStemmer("spanish")
    stemWordText=[[] for i in range(len(wordsTexts))]
    stemPalabras=[]
    for i in palabras:
        try:
            stemPalabras.append(stemmer1.stem(i))
        except IndexError:
            pass 
    i=-1
    for text in wordsTexts:
        i=i+1
        for word in text:
            try:
                stemWordText[i].append(stemmer1.stem(word))
            except IndexError:
                pass
            
    palabras=list(set(stemPalabras))
    wordsTexts=stemWordText
    return [palabras,wordsTexts]

def create_matrix(palabras,wordsTexts):
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
        #elements=list(fD1)
        for e in idColumnas:
            auxValor=fD1.get(e)
            if auxValor!=None:
                columnaAux=idColumnas.index(e)
                matriz[i][columnaAux]=auxValor
    matriz=numpy.array(matriz)
    return [matriz,idFilas,idColumnas]


def factorization(matriz):
    nmf = NMF(n_components=n_topics, random_state=1).fit(matriz)
    F=nmf.components_ #matriz de caracteristicas (filas:caracteristicas , columnas:palabras)
    W=nmf.fit_transform(matriz) #matriz de pesos (filas:textos , columnas:caracteristicas)
    return [F,W]
    
def print_topics(F,W,idFilas,idColumnas):
    with open("Topics.txt", "w") as text_file:
        for topic_idx, topic in enumerate(F):
            print("Topic {}: ".format(topic_idx), file=text_file)
            print(" ".join([idColumnas[i] for i in topic.argsort()[:-n_top_words - 1:-1]]), file=text_file) 
    with open("TextTopics.txt", "w") as text_file:
        for text, category in enumerate(W):
            print("Text #{}:".format(idFilas[text]),end=' ',file=text_file)
            print("Category #{}".format(category.argsort()[len(category)-1]),file=text_file) #tomo el mayor(esta el ultimo al ordenarlo)

a=preprocess()
s=stemming(a[0],a[1])
m=create_matrix(s[0],s[1])
res=factorization(m[0])
print_topics(res[0],res[1],m[1],m[2])

profiler.disable()
stream= open('stats.txt','w') 
ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
ps.print_stats()