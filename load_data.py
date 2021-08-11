# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext.legacy import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import pandas as pd

def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField(dtype=torch.FloatTensor)
    train_data, test_data = datasets.IMDB(split=('train', 'test'))
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    print('type train_data {}'.format(type(train_data)))
    train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter

def putTextAndLabelToCSV(fpText,fpLabel,fpOut):
    f1=open(fpText,'r')
    arrText=f1.read().strip().split('\n')
    f1.close()
    f1 = open(fpLabel, 'r')
    arrLabel = f1.read().strip().split('\n')
    f1.close()

    lstStr = ['label\ttext']
    data=[]
    for i in range(0,len(arrText)):
        itemText=arrText[i].split('\t')[1]
        itemLabel=arrLabel[i].split('\t')[1]
        strItem='{}\t{}'.format(itemLabel,itemText)
        lstStr.append(strItem)
        tup=(itemText,itemLabel)
        data.append(tup)

    f1=open(fpOut,'w')
    f1.write('\n'.join(lstStr))
    f1.close()
    # df=pd.read_csv(fpOut,delimiter='\t')
    # print('lendf {}'.format(len(df)))
    return data


def load_dataset_customize(test_sen=None):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

    """

    fopRoot = '../dataPapers/textInSPOC/mixCode_v2/step6/'
    source_folder = fopRoot
    # destination_folder + fopRoot + 'result-lstm/'

    fpTrain = fopRoot + 'train.csv'
    fpTextTrain = fopRoot + 'train.text.txt'
    fpLabelTrain = fopRoot + 'train.label.txt'

    fpValid = fopRoot + 'valid.csv'
    fpTextValid = fopRoot + 'testP.text.txt'
    fpLabelValid = fopRoot + 'testP.label.txt'

    fpTest = fopRoot + 'test.csv'
    fpTextTest = fopRoot + 'testW.text.txt'
    fpLabelTest = fopRoot + 'testW.label.txt'

    train_data=putTextAndLabelToCSV(fpTextTrain, fpLabelTrain, fpTrain)
    valid_data=putTextAndLabelToCSV(fpTextValid, fpLabelValid, fpValid)
    test_data=putTextAndLabelToCSV(fpTextTest, fpLabelTest, fpTest)
    idx_train=len(train_data)
    idx_valid=len(train_data)+len(valid_data)
    idx_test=len(train_data)+len(valid_data)+len(test_data)
    all_data=train_data+valid_data+test_data

    tokenize = lambda x: x.split()
    # TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True,
    #                   fix_length=200)
    # LABEL = data.LabelField(dtype=torch.FloatTensor)
    # train_data, test_data = datasets.IMDB(split=('train', 'test'))

    import torch
    from torchtext.legacy.data import Dataset, Example, Field
    from torchtext.legacy.data import Iterator, BucketIterator

    TEXT = Field(sequential=True, tokenize=lambda x: x.split(),
                 lower=True, use_vocab=True)
    LABEL = Field(sequential=False, use_vocab=True)



    FIELDS = [('text', TEXT), ('label', LABEL)]

    train_samples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),
                        train_data))
    valid_samples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),
                        valid_data))
    test_samples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),
                        test_data))
    all_samples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),
                        all_data))

    dt_all=Dataset(all_samples, fields=FIELDS)
    dt_train = Dataset(train_samples, fields=FIELDS)
    dt_valid = Dataset(valid_samples, fields=FIELDS)
    dt_test = Dataset(test_samples, fields=FIELDS)
    # dt_train = dt_all[:idx_train]
    # dt_valid = dt_all[idx_train:idx_valid]
    # dt_test = dt_all[idx_valid:idx_test]

    TEXT.build_vocab(dt_all, vectors="glove.6B.100d")
    LABEL.build_vocab(dt_all, vectors="glove.6B.100d")

    # TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    # LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print('str vocab {}'.format(str(TEXT.vocab[0])))
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))
    print('type train_data {}'.format(type(train_data)))
    # train_data, valid_data = train_data.split()  # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((dt_train, dt_valid, dt_test), batch_size=32,
                                                                   sort_key=lambda x: len(x.text), repeat=False,
                                                                   shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
