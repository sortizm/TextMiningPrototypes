from __future__ import print_function
from time import time
import nltk
import numpy
import matplotlib
import sklearn
from sklearn.cluster import KMeans
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.corpus import nps_chat
from nltk.corpus import stopwords

"""
Importamos el corpus con el que vamos a trabajar
"""
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
corpusdir = r"C:\Users\Ignacio\Downloads\corpusCine\corpusCriticasCine" #CHANGE THIS (local adress of the corpus)
newcorpus = PlaintextCorpusReader(corpusdir, ".*\.xml") 

## FILTERING

corpus=newcorpus #selecting the corpus to use
t0 = time() #initializing the timer
"""
We are going to create three different filters
filtro: filters the word frecuency (related to the whole of texts) and the length of words
filtro2: Keeps the words which appear in the range of a percentage (for example: if a word
        appears in less than 5% of the texts or in more than 95% of the text, it is discarded
        because is not relevant enough) 
filtro3: Keeps the words which appears more than a certain number of times in a text

The one which reduces the number of words the most is filtro2 (especially if we have a large number
of texts)

We mix the three filters altogether, but that is not necesary.
(FILTERS NEED MORE THOUGHT)
"""
print("Filtering unneccesary words...")
#palabras=sorted(set(w.lower() for w in corpus.words() if w.isalnum()))
#fD=nltk.FreqDist()
listaAux=[] #used afterwards to check that a word is in less than the 95% y at least in the 5% of the texts
wordsTexts=[]
#filtro3=[]
files=corpus.fileids()
lenFiles=len(files)

for textoId in files:
    wordsRep=list(w.lower() for w in corpus.words(textoId) if w.isalnum())
    wordsTexts.append(wordsRep)
    listaAux.append(set(wordsRep))
    #-wordsRep=(w.lower() for w in corpus.words(textoId) if w.isalnum())
    #fDaux=nltk.FreqDist(wordsRep)
    #fD |= fDaux
    #-wordsRep=(w.lower() for w in corpus.words(textoId) if w.isalnum())
    #filtro3Aux=list(w for w in fDaux if fDaux[w]>3)
    #filtro3=filtro3+ filtro3Aux
#bucle para recorrer listaAux
textosMax=numpy.round(lenFiles*95/100)
textosMin=numpy.round(lenFiles*5/100)
dicNum=dict() #diccionario que contiene las palabras y el numero de textos en los que aparece
for listas in listaAux:
    for i in listas:
        flag=True
        if i not in dicNum:
            dicNum[i]=1
            flag=False
        if i in dicNum and flag:
            dicNum[i]=dicNum.get(i)+1
            flag=False
filtro2=set(a for a,b in dicNum.items() if b>=textosMin and b<=textosMax)
#filtro=set(w for w in palabras if fD[w]>1 and len(w)>3) #tienen que estar en mas de un texto ()
#palabrasF1=filtro-set(stopwords.words('spanish'))
palabrasF2=filtro2-set(stopwords.words('spanish'))
#palabrasF3=set(filtro3)-set(stopwords.words('spanish'))
#Uno ambos criterios
palabras=sorted(palabrasF2)

print("done in %0.3fs." % (time() - t0))

## CREATING THE MATRIX

print("Creating data matrix...")
t0 = time() #initializing the timer
#Creo la matriz
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
            
print("done in %0.3fs." % (time() - t0))
## FACTORIZING

#tenemos los datos en matriz (falta aumentar el filtrado, quitar las stopwords
#quitar las que solo aprarezcan en un documento y las que aparezcan en el 95% )
print("Factorizing non-negative matrix...")
t0 = time() #initializing the timer
from sklearn.decomposition import NMF

matriz=numpy.array(matriz)
n_topics = 8
n_top_words = 10
nmf = NMF(n_components=n_topics, random_state=1).fit(matriz)
F=nmf.components_ #matriz de caracteristicas (filas:caracteristicas , columnas:palabras)
W=nmf.fit_transform(matriz) #matriz de pesos (filas:textos , columnas:caracteristicas)
print("done in %0.3fs." % (time() - t0))

## PRINT TOPICS
for topic_idx, topic in enumerate(F):
    print("Topic #%d:" % topic_idx)
    print(" ".join([idColumnas[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]])) #lo recorre desde el final con n_top_words elementos tomados
    print()
## PRINT TEXT TOPICS
#sacar a que categoria pertenece cada texto
#CAMBIAR FORMATO (categoria con los textos dentro por orden de pertenencia)
for text, category in enumerate(W):
    print("Text #%s:" %idFilas[text] )
    print("Category #%d" %category.argsort()[len(category)-1]) #tomo el mayor(esta el ultimo al ordenarlo)
    print()