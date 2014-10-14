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

corpus=gutenberg #corpus a usar
t0 = time()

print("Filtering unneccesary words...")
palabras=sorted(set(w.lower() for w in corpus.words() if w.isalnum()))
fD=nltk.FreqDist()
listaAux=[] #para comprobar que esten en menos del 95% y al menos en dos textos
for textoId in corpus.fileids():
    #wordsAux=set(w.lower() for w in corpus.words(textoId) if w.isalnum())
    #wordsAux=wordsAux-set(stopwords.words('english'))
    wordsRep=(w.lower() for w in corpus.words(textoId) if w.isalnum())
    listaAux.append(set(wordsRep))
    wordsRep=(w.lower() for w in corpus.words(textoId) if w.isalnum())
    fDaux=nltk.FreqDist(wordsRep)
    fD |= fDaux
#bucle para recorrer listaAux
textosMax=numpy.round(len(corpus.fileids())*95/100)
textosMin=2
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
filtro=set(w for w in palabras if fD[w]>1 and len(w)>1) #tienen que estar en mas de un texto ()
palabrasF1=filtro-set(stopwords.words('english'))
palabrasF2=filtro2-set(stopwords.words('english'))
#Uno ambos criterios
palabras=sorted(palabrasF1&palabrasF2)

#Creo la matriz
ficheros=corpus.fileids()
filas=len(corpus.fileids()) #numero de filas de la matriz
columnas=len(palabras)
print("done in %0.3fs." % (time() - t0))

print("Creating data matrix...")
idFilas=[]
idColumnas=[]
matriz=[[0]*columnas for i in range(filas)]
for i in range(columnas) :
    idColumnas.append(palabras[i])
for i in range(filas):
    idFilas.append(ficheros[i])
    
for textoId in corpus.fileids():
    wordsAux=(w.lower() for w in corpus.words(textoId) if w.isalnum())
    fD=nltk.FreqDist(wordsAux)
    filaAux=idFilas.index(textoId)
    while(len(fD)>0):
        tupla=fD.popitem()
        if  tupla[0] in idColumnas:
            columnaAux=idColumnas.index(tupla[0])
            matriz[filaAux][columnaAux]=tupla[1]
print("done in %0.3fs." % (time() - t0))

#tenemos los datos en matriz (falta aumentar el filtrado, quitar las stopwords
#quitar las que solo aprarezcan en un documento y las que aparezcan en el 95% )
print("Factorizing non-negative matrix...")
from sklearn.decomposition import NMF

matriz=numpy.array(matriz)
n_topics = 3
n_top_words = 20
nmf = NMF(n_components=n_topics, random_state=1).fit(matriz)
F=nmf.components_ #matriz de caracteristicas (filas:caracteristicas , columnas:palabras)
W=nmf.fit_transform(matriz) #matriz de pesos (filas:textos , columnas:caracteristicas)
print("done in %0.3fs." % (time() - t0))

for topic_idx, topic in enumerate(F):
    print("Topic #%d:" % topic_idx)
    print(" ".join([idColumnas[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]])) #lo recorre desde el final con n_top_words elementos tomados
    print()

#sacar a que categoria pertenece cada texto
for text, category in enumerate(W):
    print("Text #%s:" %idFilas[text] )
    for i in range(category.size):
        if category.item(i)>1:
            print("Category #%d" % i)
    print()