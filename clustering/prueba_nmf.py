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
#from scipy import sparse
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
#C:\Users\Ignacio\Desktop\corpusTweets
#C:\Users\Ignacio\Downloads\corpusPrueba
#C:\Users\Ignacio\Downloads\corpusCine\corpusTxt


from nltk.corpus.reader.plaintext import PlaintextCorpusReader
corpusdir = r"C:\Users\Ignacio\Downloads\corpusCine\corpusTxt" #CHANGE THIS (local adress of the corpus)
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


def preprocess():
    """
    Filters the words in the corpus. We discard numbers, very short words and words appearing in less than 5% of the texs 
    and in more than the 95% of the texts. (NEED TO BE CHANGED ---> the percentage needs to change depending on the number of texts)
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
    """
    Reduces the amount of filtering words keeping only their stems
    return: a list with the words and the texts filtered
    CAN BE IMPROVED USING A SPARSE MATRIX
    """
    if ('stemmed_words' not in datavars or 'stemmed_texts' not in datavars) and os.path.exists(FILENAME):
        with open(FILENAME, 'r') as datafile:
            datavars.update(json.load(datafile))
    if 'stemmed_words' not in datavars or 'stemmed_texts' not in datavars: # even after loading file
        print('No json found in the stemming')
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
        datavars['stemmed_words'] = palabras
        datavars['stemmed_texts'] = wordsTexts
        with open(FILENAME, 'w') as datafile:
            json.dump(datavars, datafile, indent=4)
            

    return [datavars['stemmed_words'],datavars['stemmed_texts']]


def create_matrix(palabras,wordsTexts):
    """
    Creates a matrix with the count of each word. The matrix rows are the words and the matrix columns are the corpus files.
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
        #elements=list(fD1)
        for e in idColumnas:
            auxValor=fD1.get(e)
            if auxValor!=None:
                columnaAux=idColumnas.index(e)
                matriz[i][columnaAux]=auxValor
    matriz=numpy.array(matriz)
    return [matriz,idFilas,idColumnas]


def idf(matriz):
    """
    Transform the word frequencies in the matrix to reflect to add information using the TF*IDF heuristic
    return: the modified matrix
    """

    #could be improved using sparse matrix
    #matriz=sparse.csr_matrix(matriz)
    filas=matriz.shape[0] #number of files
    columnas=matriz.shape[1] #number of words
    lista_aux=[None]*columnas
    for j in range(columnas):
        count=0
        for i in range(filas):
           if matriz[i][j]!=0:
               count=count+1
        lista_aux[j]=count    
    
    idf_matriz= numpy.empty([filas,columnas],dtype=numpy.float64)
    for i in range(filas):
        for j in range(columnas):
            idf_matriz[i][j]= numpy.log(matriz[i][j]+1) * numpy.log(filas/lista_aux[j])
                
    return idf_matriz
    
  
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
a=preprocess()
print("done in %0.3fs." % (time() - t0))
print("Stemming")
s=stemming(a[0],a[1])
print("done in %0.3fs." % (time() - t0))
print("Creating data matrix")
m=create_matrix(s[0],s[1])
print("done in %0.3fs." % (time() - t0))
print("Converting matrix values to TF*IDF")
f=idf(m[0])
print("done in %0.3fs." % (time() - t0))
print("Non-negative factorization")
res=factorization_NMF(f)
print("done in %0.3fs." % (time() - t0))
print("Printing results")
print_topics(res[0],res[1],m[1],m[2])
print("done in %0.3fs." % (time() - t0))

profiler.disable()
stream= open('stats.txt','w') 
ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
ps.print_stats()
stream.close()