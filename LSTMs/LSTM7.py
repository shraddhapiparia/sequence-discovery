from gensim.models import Word2Vec
import collections
import operator
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Activation, LSTM, Embedding, GRU , Conv1D, MaxPooling1D, Conv2D, Convolution1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer

sentences = []
vocab = []
vector_dim = 7
xtrain = []
ytrain = []
xtest = []
ytest = []

# Reading input data
print "Reading training data"
f = open('combined_trainfile.txt',"r")
input_text = f.read()
for line in input_text.split('\n'):
    if len(line) > 1:
	action = line
	sentences.append([action])
	vocab.append(action)

vocab_size = len(vocab)

# Training input data on Word2Vec model
print "Training model"
model = Word2Vec(sentences, size = 7, min_count = 1)
model.save

# Using Word2Vec model to generate embeddings 
embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
for i in range(len(model.wv.vocab)):
    embedding_vector =  model.wv[model.wv.index2word[i]]
    #print "val is: ", model.wv.index2word[i]
    if embedding_vector is not None:
	embedding_matrix[i] = embedding_vector

tokenizer = Tokenizer(num_words = 41)

#print len(embedding_matrix)

print "Reading from train file"
with open('combined_trainfile.txt','r') as train_file:
    lines = train_file.readlines()
    for i in range(0,len(lines)-7):
	x1 = model.wv.vocab[lines[i].strip('\n')].index
	x2 = model.wv.vocab[lines[i+1].strip('\n')].index
	x3 = model.wv.vocab[lines[i+2].strip('\n')].index
	x4 = model.wv.vocab[lines[i+3].strip('\n')].index
	x5 = model.wv.vocab[lines[i+4].strip('\n')].index
	x6 = model.wv.vocab[lines[i+5].strip('\n')].index
	x7 = model.wv.vocab[lines[i+6].strip('\n')].index
	x8 = model.wv.vocab[lines[i+7].strip('\n')].index
	xtrain.append([x1,x2,x3,x4,x5,x6,x7])
	ytrain.append([x8])

print "Reading from test file"
with open('combined_testfile.txt','r') as test_file:
    lines = test_file.readlines()
    for i in range(0,len(lines)-7):
	x1 = model.wv.vocab[lines[i].strip('\n')].index
	x2 = model.wv.vocab[lines[i+1].strip('\n')].index
	x3 = model.wv.vocab[lines[i+2].strip('\n')].index
	x4 = model.wv.vocab[lines[i+3].strip('\n')].index
	x5 = model.wv.vocab[lines[i+4].strip('\n')].index
	x6 = model.wv.vocab[lines[i+5].strip('\n')].index
	x7 = model.wv.vocab[lines[i+6].strip('\n')].index
	x8 = model.wv.vocab[lines[i+7].strip('\n')].index
	xtest.append([x1,x2,x3,x4,x5,x6,x7])
	ytest.append([x8])
'''
	for j in range(len(model.wv.vocab)):
	    if model.wv.index2word[j] == lines[i].strip('\n'):
		x = embedding_matrix[j]
	for k in range(len(model.wv.vocab)):
	    if model.wv.index2word[k] == lines[i+1].strip('\n'):
		y = embedding_matrix[k]
	for k in range(len(model.wv.vocab)):
	    if model.wv.index2word[k] == lines[i+2].strip('\n'):
		z = embedding_matrix[k]
	xtrain.append([x,y])
	ytrain.append([z])
	#print 'Index is ',model.wv.vocab[tr_line].index

#print xtrain[0]


X_train = np.asarray(xtrain)
Y_train = np.asarray(ytrain)'''
'''
X_train = tokenizer.sequences_to_matrix(xtrain,mode='binary')
Y_train = tokenizer.sequences_to_matrix(ytrain,mode='binary')
'''
#print X_train.shape , embedding_matrix.shape

X_train = np.asarray(xtrain)
Y_train = tokenizer.sequences_to_matrix(ytrain,mode='binary')

X_test = np.asarray(xtest)
Y_test = tokenizer.sequences_to_matrix(ytest,mode='binary')

#embedding_matrix = embedding_matrix[1:]
#print "X_train is: ",X_train[0]#, ", Y_train is: ", Y_train'''

# Build the sequential neural network
print "Building network"
model_nn = Sequential()
e = Embedding(41,7,input_length = 7, weights=[embedding_matrix], trainable = True)
#e = Embedding(31,output_dim = embedding_matrix.shape[1],weights = [embedding_matrix],input_length=embedding_matrix.shape[0]) 
model_nn.add(e)
#model_nn.add(Input(shape=(2,)))
'''
model_nn.add(Conv1D(filters=5, kernel_size=2, activation='relu', padding='valid'))
model_nn.add(MaxPooling1D(pool_length=1))
model_nn.add(Flatten())
'''
model_nn.add(LSTM(42, dropout=0.5, recurrent_dropout=0.5))
model_nn.add(Activation('relu'))
#model_nn.add(LSTM(128, dropout=0.7))
model_nn.add(Dense(41, activation='softmax'))
model_nn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','categorical_accuracy'])

plot_model(model_nn, to_file="LSTM.png", show_shapes=True)

earlystopping = EarlyStopping(monitor='val_loss', patience=20)
print "Fit data", Y_train.shape
model_nn.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=100, verbose = 1,callbacks=[earlystopping])

score = model_nn.evaluate(X_test,Y_test,batch_size=1,verbose = 0)
preds = model_nn.predict(X_test,verbose= 0)
pred_classes = model_nn.predict_classes(X_test,verbose = 0)

print ("shape is ",pred_classes.shape)
print ("shape of preds is ",preds.shape)


print('Test score:', score[0])
print('Test Accuracy:', score[1])
print('Test Categorical accuracy:', score[2])





 


