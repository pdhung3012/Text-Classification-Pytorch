import os
import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))

from operator import itemgetter
from numpy import unique


def preprocessText(strInput,stop_words,ps,lemmatizer):
    word_tokens = word_tokenize(strInput)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    strTemp=' '.join(filtered_sentence)

    words=word_tokenize(strTemp)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)
    strOutput=strTemp
    return strOutput

def preprocessTextV2(strInput,ps,lemmatizer):
    words=word_tokenize(strInput)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)
    strOutput=strTemp
    return strOutput


def preprocessTextV3(strInput,ps,lemmatizer):
    words=word_tokenize(strInput)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)

    wordsList = nltk.word_tokenize(strTemp)
    tagged = nltk.pos_tag(wordsList)
    # print('tagged {}'.format(type(tagged[0][0])))
    lstContentI = []
    for it in tagged:
        strIt = '{} {}'.format(it[0], it[1])
        lstContentI.append(strIt)
    strTemp=' '.join(lstContentI)

    strOutput=strTemp
    return strOutput

def preprocessTextV3_FilerWord(strInput,setSparsityWords,ps,lemmatizer):
    words=word_tokenize(strInput)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)

    wordsList = nltk.word_tokenize(strTemp)
    tagged = nltk.pos_tag(wordsList)
    # print('tagged {}'.format(type(tagged[0][0])))
    lstContentI = []
    for it in tagged:
        if(it[0] in setSparsityWords):
            continue
        strIt = '{} {}'.format(it[0], it[1])
        lstContentI.append(strIt)
    strTemp=' '.join(lstContentI)

    strOutput=strTemp
    return strOutput

def preprocessTextV3_FilerWordAndReplace(strInput,setSparsityWords,ps,lemmatizer):
    words=word_tokenize(strInput)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)

    wordsList = nltk.word_tokenize(strTemp)
    tagged = nltk.pos_tag(wordsList)
    # print('tagged {}'.format(type(tagged[0][0])))
    lstContentI = []
    for it in tagged:
        if(it[0] in setSparsityWords):
            strIt = '{} {}'.format('NN', 'REMOVE_WORD')
        else:
            strIt = '{} {}'.format(it[0], it[1])
        lstContentI.append(strIt)
    strTemp=' '.join(lstContentI)

    strOutput=strTemp
    return strOutput


def preprocessTextV4(strInput,ps,lemmatizer):
    words=word_tokenize(strInput)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)

    wordsList = nltk.word_tokenize(strTemp)
    tagged = nltk.pos_tag(wordsList)
    # print('tagged {}'.format(type(tagged[0][0])))
    lstContentI = []
    lstStr1=[]
    lstStr2 = []
    for it in tagged:
        lstStr1.append(it[0])
        lstStr2.append(it[1])

    strTemp=' '.join([' '.join(lstStr1),' '.join(lstStr2)])
    strOutput=strTemp
    return strOutput

def preprocessFilterOnlyVerbNoun(strInput,ps,lemmatizer):
    words=word_tokenize(strInput)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)

    wordsList = nltk.word_tokenize(strTemp)
    tagged = nltk.pos_tag(wordsList)
    # print('tagged {}'.format(type(tagged[0][0])))
    lstContentI = []
    lstStr1=[]
    lstStr2 = []
    for it in tagged:

        strForm=str(it[1])
        # or strForm.startswith('NN')
        if strForm.startswith('V') or strForm.startswith('NN') or strForm=='.':
            # print(it)
            lstStr1.append(it[0])
            # lstStr2.append(it[1])

    strTemp=' '.join(lstStr1)
    # print(strTemp)
    strOutput=strTemp
    return strOutput

def preprocessFilterOnlyVerbNoun(strInput,ps,lemmatizer):
    words=word_tokenize(strInput)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)

    wordsList = nltk.word_tokenize(strTemp)
    tagged = nltk.pos_tag(wordsList)
    # print('tagged {}'.format(type(tagged[0][0])))
    lstContentI = []
    lstStr1=[]
    lstStr2 = []
    for it in tagged:

        strForm=str(it[1])
        # or strForm.startswith('NN')
        if strForm.startswith('V') or strForm.startswith('NN') or strForm=='.':
            # print(it)
            lstStr1.append(it[0])
            # lstStr2.append(it[1])

    strTemp=' '.join(lstStr1)
    # print(strTemp)
    strOutput=strTemp
    return strOutput

import time
# https://medium.com/swlh/nlp-text-preprocessing-techniques-ea34d3f84de4
def preprocessFollowingNLPStandard(strInput,ps,lemmatizer):
    start_time = time.time()
    strStep1=strInput.replace('\t',' ').replace('\n',' ').strip()
    strLower=strStep1.lower().replace('\t',' ').replace('\n',' ').strip()
    words=word_tokenize(strLower)
    lstStems=[]
    for w in words:
        lstStems.append(ps.stem(w))
    strTemp=' '.join(lstStems)

    words = word_tokenize(strTemp)
    lstLems = []
    for w in words:
        lstLems.append(lemmatizer.lemmatize(w))
    strTemp = ' '.join(lstLems)

    wordsList = nltk.word_tokenize(strTemp)
    tagged = nltk.pos_tag(wordsList)
    # print('tagged {}'.format(type(tagged[0][0])))
    lstContentI = []
    lstStr1=[]
    lstStr2 = []
    for it in tagged:

        strForm=str(it[1]).strip()
        strText = str(it[0]).strip()
        # or strForm.startswith('NN')
        if (not strText=='') and (not strForm==''):
            # print(it)
            lstStr1.append(it[0])
            lstStr2.append(it[1])

    strPre=' '.join(lstStr1)
    strPOS=' '.join(lstStr2)
    # print(strTemp)
    end_time = time.time()
    running_time = end_time - start_time
    return strStep1,strPre,strPOS,running_time


def filterGapLabels(y_expected,y_predicted,percentRemove):
    lstGaps=[]
    y_expected=y_expected.tolist()
    y_predicted=y_predicted.tolist()
    for index in range(1,len(y_expected)):
        # print('item {}'.format(y_expected[index]))
        minus=abs(y_expected[index]-y_predicted[index])
        newTuple=(index,minus)
        lstGaps.append(newTuple)
    sorted(lstGaps, key=itemgetter(1),reverse=True)
    numberRemove=int(percentRemove*len(lstGaps))
    for index in range(0,numberRemove):
        lstGaps.pop(0)

    lstMaintained=[]
    for index in range(0,len(lstGaps)):
        # print(lstGaps[index][0])
        lstMaintained.append(lstGaps[index][0])

    newy_expected=[]
    newy_predicted=[]
    for index in range(1,len(y_predicted)):
        if index in lstMaintained:
            newy_predicted.append(y_predicted[index])
            newy_expected.append(y_expected[index])
    return newy_expected,newy_predicted










def initDefaultTextEnvi():
    nlp_model = spacy.load('en_core_web_sm')
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    return nlp_model,nlp

def getSentences(text,nlp):
    result=None
    try:
        document = nlp(text)
        result= [sent.string.strip() for sent in document.sents]
    except Exception as e:
        print('sone error occured {}'.format(str(e)))
    return result


def preprocess(textInLine):
    text = textInLine.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word in words]
    # doc = [word for word in doc if word.isalpha()]
    return ' '.join(doc)



# Python program to illustrate the intersection
# of two lists in most simple way
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
def diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def sortTuple(tup,isReverse):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: x[1],reverse=isReverse)
    return tup

def getNdArray(inputTrainX,inputTrainY):
    listTrainX=[]

    for i in range(0,len(inputTrainX)):
        item=inputTrainX.iloc[i]
        listTrainX.append(item)

    listTrainY=inputTrainY.tolist()
    arrayTrainX= np.array(listTrainX)
    arrayTrainY=np.array(listTrainY)
    return arrayTrainX,arrayTrainY


def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.makedirs(fopOutput, exist_ok=True)
        #print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")

def scoreName(val):
    text=0
    if val <= 5:
        text = 0
    elif val>5 and val<=15:
        text = 1
    elif val>15 and val<=40:
        text = 2
    else:
        text = 3
    return text
