#deal with tensors
import torch

#handling text data
from torchtext.legacy import data
# from torchtext.legacy.data import *
#Reproducing same results
SEED = 2019

#Torch
torch.manual_seed(SEED)

def putTextAndLabelToCSV(fpText,fpLabel,lstText,fpOut):
    f1=open(fpText,'r')
    arrText=f1.read().strip().split('\n')
    f1.close()
    f1 = open(fpLabel, 'r')
    arrLabel = f1.read().strip().split('\n')
    f1.close()

    lstStr = ['label,text']
    data=[]
    for i in range(0,len(arrText)):
        itemText=arrText[i].split('\t')[1].replace(',',' COMMA ')
        itemLabel=arrLabel[i].split('\t')[1].replace(',',' COMMA ')
        strItem='{},{}'.format(itemLabel,itemText)
        lstStr.append(strItem)
        tup=(itemText,itemLabel)
        data.append(tup)
        lstText.append(itemText)

    f1=open(fpOut,'w')
    f1.write('\n'.join(lstStr))
    f1.close()
    # df=pd.read_csv(fpOut,delimiter='\t')
    # print('lendf {}'.format(len(df)))
    return data


fopRoot='../dataPapers/'
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

lstTextAll=[]
fpTextAll = fopRoot + 'text8'
train_data = putTextAndLabelToCSV(fpTextTrain, fpLabelTrain,lstTextAll, fpTrain)
valid_data = putTextAndLabelToCSV(fpTextValid, fpLabelValid,lstTextAll, fpValid)
test_data = putTextAndLabelToCSV(fpTextTest, fpLabelTest,lstTextAll, fpTest)

f1=open(fpTextAll,'w')
f1.write('\n'.join(lstTextAll))
f1.close()

idx_train = len(train_data)
idx_valid = len(train_data) + len(valid_data)
idx_test = len(train_data) + len(valid_data) + len(test_data)
all_data = train_data + valid_data + test_data


#Cuda algorithms
torch.backends.cudnn.deterministic = True
TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True,use_vocab=True)
fields = [('label', LABEL),('text',TEXT)]
#loading custom dataset
train_data=data.TabularDataset(path =fpTrain,format = 'csv',fields = fields,skip_header = True)
valid_data=data.TabularDataset(path =fpValid,format = 'csv',fields = fields,skip_header = True)
test_data=data.TabularDataset(path =fpTest,format = 'csv',fields = fields,skip_header = True)

#print preprocessed text
print(vars(train_data.examples[0]))
import random
# train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))
#initialize glove embeddings
# TEXT.build_vocab(train_data,valid_data,test_data,vectors = "glove.6B.100d")
from torchtext.vocab import Vectors
vectors = Vectors(name='vectors_glove.txt', cache=fopRoot)
TEXT.build_vocab(train_data,valid_data,test_data,vectors = vectors)
LABEL.build_vocab(train_data,valid_data,test_data)

#No. of unique tokens in text
print("Size of TEXT vocabulary:",len(TEXT.vocab))

#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))

#Commonly used words
print(TEXT.vocab.freqs.most_common(10))

#Word dictionary
print(TEXT.vocab.stoi)
#check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#set batch size
BATCH_SIZE = 64

#Load an iterator
train_iterator, valid_iterator ,test_iterator= data.BucketIterator.splits(
    (train_data, valid_data,test_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True,
    device = device)

import torch.nn as nn


class classifier(nn.Module):

    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs

#define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

#instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers,
                   bidirectional = True, dropout = dropout)

# architecture
print(model)


# No. of trianable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# Initialize the pretrained embedding
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

import torch.optim as optim

# define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()


# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# push to cuda if available
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for batch in iterator:
        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text
        text_lengths=text_lengths.cpu()

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        loss = criterion(predictions, batch.label)

        # compute the binary accuracy
        acc = binary_accuracy(predictions, batch.label)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text
            text_lengths = text_lengths.cpu()

            # convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    # train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    # evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), fopRoot+'saved_weights.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

#load weights
path=fopRoot+'/saved_weights.pt'
model.load_state_dict(torch.load(path));
model.eval();

#inference
import spacy
nlp = spacy.load('en_core_web_sm')

def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction
    return prediction.item()

# #make predictions
# predict(model, "Are there any sports that you don't like?")
#
# #insincere question
# predict(model, "Why Indian girls go crazy about marrying Shri. Rahul Gandhi ji?")

test_loss, test_acc = evaluate(model, test_iterator, criterion)
print('test loss and test acc \n{}\t{}'.format(test_loss,test_acc))