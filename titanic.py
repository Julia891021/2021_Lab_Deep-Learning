# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')
features = df[['Pclass','Sex','Age','SibSp','Parch','Embarked']]
target = df['Fare']
#print(features.head())
#print(target.head())
#%%
#統計各個空欄位數量
print(features.isnull().sum())  
print('---------------------------')
print(target.isnull().sum())
print('---------------------------')
#以平均數填 也可最小值或其他
age_mean = features['Age'].mean()
fare_mean = target.mean()
#填補空値
features['Age'].fillna(age_mean, inplace = True)
target.fillna(fare_mean, inplace = True)
#%%
#normalize
#standard 
#min-max
def minMax_normalize(data):
    return(data-data.min()/(data.max()-data.min()))
def std_normalize(data):
    return (data-data.mean()/data.std)
features['Age'] = minMax_normalize(features['Age'] )
features['SibSp'] = minMax_normalize(features['SibSp'] )
features['Parch'] = minMax_normalize(features['Parch'] )

# print(features['Age'].max())
# print(features['SibSp'].max())
# print(features['Parch'].max())
# print()
# print(features['Age'].min())
# print(features['SibSp'].min())
# print(features['Parch'].min())

#%%
#類別型資料轉數值型
# print(features['Sex'].unique)
# print(features['Pclass'].unique)
# print(features['Embarked'].unique)

features['Sex'] = features['Sex'].map({'female':0, 'male':1})
features['Pclass'] = features['Pclass']-1
features['Embarked'] = features['Embarked'].map({'C':0, 'Q':1, 'S':2})
#透過one-hot encoding 類似擴增 EX:Pclass有三類 變成PclassA PclassB PclassC 以1 0 表示是否有   轉換資料 Method1
from keras.utils import to_categorical
pclass = pd.DataFrame(to_categorical(features['Pclass'], num_classes= 3, dtype = int)) #num_class 資料有幾種類別
embarked = pd.DataFrame(to_categorical(features['Embarked'], num_classes= 3, dtype = int)) 
#concat dataframe
features = pd.concat([features[['Sex', 'Age','SibSp','Parch']], pclass, embarked], axis =1)

#method 2 
# features = pd.get_dummies(features, columns = ['Sex'], prefix = 'sex', drop_first=True)
# features = pd.get_dummies(features, columns = ['Embarked'], prefix = 'embarked')
# features = pd.get_dummies(features, columns = ['Pclass'], prefix = 'pclass')
#print(features.head())



#%%
#切分訓練/測試資料
X = features.values
Y = target.values



x_train = X[:1000]
y_train = Y[:1000]
x_test = X[1000:]
y_test = Y[1000:]

#%%
#增加suffle打亂 但features target是分開的 所以要固定再suffle
#要suffle原因 : 因為不知道原本資料是否有排序
np.random.seed(2021) #固定random結果()數字可以改變 若要執行兩次random每次random前都要random.seed() 
indices = np.random.permutation(x_train.shape[0])  #x_train.shape() -> (1309,12) x_train.shape[0]->1309 此為以1309為range打亂
#print(indices)

X = X[indices]
Y = Y[indices]
#%%
from keras.models import Model
from keras.layers import Dense, Input

#建立model
x = Input(shape = (10,))  #有10個input  #(x)代表接第一層
h = Dense(units = 10, activation= 'relu')(x)  #一層裡面有10個neron parameter:110 因為10*10 weight + 10 bias
h = Dense(units = 5, activation = 'relu')(h)  #parameter 55
o = Dense(units = 1, activation = 'relu')(h)  #parameter 6

#%%
model = Model(inputs = x,  outputs = o)
model.compile(loss = 'MSE', optimizer ='adam')
model.summary()

train_history = model.fit(
        x = x_train,
        y = y_train,
        validation_split=0.2, #將training 切出 validation set  #將training data中的0.2部分不當training 用來當驗證集 
        batch_size = 50, #每次訓練幾筆資料再更新
        epochs = 100, # 要反覆訓練幾次
        verbose = 1)  #訓練紀錄顯示的方法（0 - 不顯示、1 - 完整顯示、2 - 精簡顯示）
#%%
import matplotlib.pyplot as plt
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
plt.close()

#%%
test_loss = model.evaluate(x=x_test,y=y_test)
y_predict = model.predict(x_test).reshape((-1))   #reshape(-1)代表變成一維

# print(test_loss)
# print(y_predict[0])
# print(y_test[0])

#%%
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

# plt.imshow(x_train[0])
# plt.show()
# plt.imshow(x_train[1])
# plt.show()


def preprocess(x, y):
    mask_6 = (y ==6)  #若是6 -> True
    mask_8 = (y == 8)
    mask = np.logical_or(mask_6, mask_8)  
    #利用mask去過濾想要的資訊
    
    x = x[mask] #只取6 or 8
    x = x.reshape((x.shape[0], -1)) #將二維的圖片變成向量 #x.shape[0]是1932 -1代表把後面剩下的變成一維 例如28*28就不用 管它長怎樣了 直接變一維
    
    y[mask_6] = 0
    y[mask_8] = 1
    y = y[mask]
    
    return x, y

def normalize(x):
    return x/255

x_test, y_test = preprocess(x_test, y_test)
x_train, y_train = preprocess(x_train, y_train)

#%%  
#標準化
x_test = normalize(x_test)
y_test = normalize(y_test)
#%%
#Build model
from keras.models import Sequential
from keras.layers import Dense
#也可以用一層一層的方式看自己寫法
model = Sequential()
model.add(Dense(units =10, input_dim  = x_train.shape[1],activation= 'sigmoid'))
model.add(Dense(units = 1, activation='sigmoid'))
#model.add(Dense(units = 1, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])

train_history = model.fit(
        x = x_train,
        y = y_train,
        validation_split=0.2,
        batch_size = 500, #每次訓練幾筆資料再更新
        epochs = 50, # 要反覆訓練幾次
        verbose = 1)  #訓練紀錄顯示的方法（0 - 不顯示、1 - 完整顯示、2 - 精簡顯示）
#%%
#Evaluation Loss
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
plt.close()
#%%
#Accuracy
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()
plt.close()

#%%
y_predict = model.predict(x_test).reshape((-1))
y_predict[y_predict >= 0.5] = 1
y_predict[y_predict < 0.5] = 0
y_predict = y_predict.astype('uint8')
test_acc = np.sum(y_test == y_predict) / len(y_test)
print(f'Test ACC: {test_acc}')
y_test = y_test.astype('uint8')
#%%
def confusion_matrix(y_true, y_pred):
    mat = np.zeros((2,2)) #開一個2*2 matrix
    N = len(y_true)
    for i in range(N):
        mat[y_pred[i]][y_true[i]] += 1
    return mat

print(confusion_matrix(y_test, y_predict))
