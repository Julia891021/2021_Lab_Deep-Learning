# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 14:43:17 2021

@author: Julia
"""

from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils

(X_train, Y_train),(X_test, Y_test) = cifar10.load_data()
#(Y_test的1是label)

#%%
#build the input vector
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')

#/255 ->標準化每個pixel都是255 要做min_max
X_train/=255
X_test/=255
#要標準化幫助資料收斂 以及 影響都是一樣的

#one-hot encoding of label 分類
num_class = 10
Y_train = np_utils.to_categorical(Y_train, num_class)
Y_test = np_utils.to_categorical(Y_test, num_class)
#%%
#build model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

model = Sequential(name = 'CNN_MNIST')
model.add(Conv2D(filters = 50,              
                 kernel_size = (5, 5), 
                 activation = 'relu', 
                 input_shape = (32, 32, 3))) #(28, 28, 10)
model.add(MaxPool2D(pool_size=(2, 2)))  #(14, 14, 10)

model.add(Conv2D(filters = 80,             
                 kernel_size = (5, 5), 
                 activation = 'relu', 
                 input_shape = (14, 14, 3)))  #(10, 10, 20)
model.add(MaxPool2D(pool_size=(2, 2)))  #(5, 5, 20)

model.add(Flatten())
model.add(Dense(500, activation='relu'))  #parameter = 100*(5*5*20)+100
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) #parameter = 100*10+10
model.summary()
#%%
#train model
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
train_history  = model.fit(X_train, Y_train,
                           batch_size=128,
                           epochs=10,
                           validation_data=(X_test, Y_test))
#%%
#plot accuracy
import matplotlib.pyplot as plt
plt.plot(train_history.history['accuracy'], label = 'training set')
plt.plot(train_history.history['val_accuracy'], label = 'validation set')
plt.title('Training Records')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc = 'best')
plt.show()
#plot loss
plt.plot(train_history.history['loss'], label = 'training set')
plt.plot(train_history.history['val_loss'], label = 'validation set')
plt.title('Training Records')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc = 'best')
plt.show()
#%%
import numpy as np
#predict = 吐10個 每個可能的機率
#predict.classes 只吐最有可能的那個
#繪製預測結果與答案對照
predict = model.predict_classes(X_test)
for i in range(10):
    title = 'ans = {}, pre = {}'.format(np.argmax(Y_test[i], axis = 0), predict[i])
    figure = plt.subplot(2, 5, i+1)
    figure.imshow(X_test[i].reshape(32,32,3), cmap = 'gray')
    figure.set_title(title, fontsize = 10)
    figure.set_xticks([])
    figure.set_yticks([])
    
plt.tight_layout()
plt.show()
    
#%%
#繪製混淆矩陣
import pandas as pd
print(pd.crosstab(np.argmax(Y_test, axis =1), predict, rownames=['answer'], colnames=['predict']))
#如果資料集集中 要記得shuffle 
