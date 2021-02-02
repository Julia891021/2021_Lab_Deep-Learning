# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:44:54 2021

@author: Julia
"""
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN,LSTM
from keras.layers import GRU

model = Sequential()
model.add(SimpleRNN(10,input_shape=(100,1),return_sequences=False))
model.add(Dense(units=1,activation='sigmoid'))
model.add(Dense(units=2,activation='sigmoid'))
model.summary()

model = Sequential()
model.add(LSTM(10,input_shape=(100,1),return_sequences=False))
model.add(Dense(units=1,activation='sigmoid'))
model.add(Dense(units=2,activation='sigmoid'))
model.summary()

model = Sequential()
model.add(GRU(10,input_shape=(100,1),return_sequences=False,reset_after=False))
#若沒有加reset_after會把30個bias加入(多一組bias的數量)
model.add(Dense(units=1,activation='sigmoid'))
model.add(Dense(units=2,activation='sigmoid'))
model.summary()

#%%
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN,LSTM
from keras.layers import GRU

x_train = np.load('./X_train.npy')
x_test = np.load('./X_test.npy')
y_train = np.load('./y_train.npy')
y_test = np.load('./y_test.npy')

#%%
#build model
model = Sequential()
#simple RNN
model.add(SimpleRNN(10,input_shape=(100,1),return_sequences=False))
model.add(Dense(units=1,activation='tanh'))
#model.add(Dense(units=2,activation='sigmoid'))
model.summary()
#compile
model.compile(loss = 'mse', optimizer = 'adam')
#fit
h = model.fit(x_train, 
              y_train, 
              batch_size= 16, 
              epochs = 10, 
              validation_split=0.2)
#%%

import matplotlib.pyplot as plt
#plot model loss
plt.plot(h.history['loss'], label = 'training_loss')
plt.plot(h.history['val_loss'], label = 'val_loss')
#圖例
plt.legend(loc = 'upper right') 
plt.show()
plt.close()
#%%
pred = model.predict(x_test)
#%%
plt.plot(pred, label='true')
plt.plot(y_test, label='true')
plt.legend(loc = 'upper right') 
plt.show()
plt.close()
#%%
#stock prediction
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN,LSTM,GRU
#讀檔+切割
df = pd.read_csv('SPY.csv')
features = df[['Open','High','Low','Close']]
# N = int(df.shape[0]*0.8//1)
train = features[:5076]
test = features[5076:]
#%%
#正規化
#wrong->要分別取train中的每個max值
Max = train.max()
Min = train.min()
def minMax_normalize(data):
    return(data-Min/(Max-Min))
#test要用train的min max 因為的min max照理來說不會知道test的min max
train = minMax_normalize(train)
test = minMax_normalize(test)
#%%
#建立時間序列    
x_train = np.zeros((train.shape[0]-30,30,4))  #4代表取index4
x_test = np.zeros((test.shape[0]-30,30,4))
y_train = np.zeros((train.shape[0]-30,1))
y_test = np.zeros((test.shape[0]-30,1))

for i in range(x_train.shape[0]):
    x_train[i] = train.iloc[i:i+30]
    y_train[i] = train.iloc[i+30][3]

for i in range(x_test.shape[0] - 30):
    x_test[i] = test.iloc[i:i+30]
    y_test[i] = test.iloc[i+30][3]
#%%
#simple RNN
model.add(SimpleRNN(10,input_shape=(100,1),return_sequences=False))
model.add(Dense(units=1,activation='tanh'))
#model.add(Dense(units=2,activation='sigmoid'))
model.summary()
#compile
model.compile(loss = 'mse', optimizer = 'adam')
#fit
h = model.fit(x_train, 
              y_train, 
              batch_size= 16, 
              epochs = 10, 
              validation_split=0.2)
#%%
import matplotlib.pyplot as plt
#plot model loss
plt.plot(h.history['loss'], label = 'training_loss')
plt.plot(h.history['val_loss'], label = 'val_loss')
#圖例
plt.legend(loc = 'upper right') 
plt.show()
plt.close()
#%%
pred = model.predict(x_test)
plt.plot(pred, label='true')
plt.plot(y_test, label='true')
plt.legend(loc = 'upper right') 
plt.show()
plt.close()
    