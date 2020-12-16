import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.optimizers import Adam


DATADIR = "C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem1/Data/train_set"
CATEGORIES = ['Burnt_Pizza','Good_Pizza']

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        
print(img_array)
print(img_array.shape)

IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in (os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
            
create_training_data()

print(len(training_data))

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)

X_train = X/255.0
number_of_classes = 2
y2 = np.array(y, dtype='uint8')

model = Sequential()
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(5, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, y2, validation_data=(X_train, y2), epochs=50, batch_size=100)
model.summary()

model.save('C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem1/l2p1_9746.model')

##################################################################################################

test_data = []
DAT = "C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem1/Data"
CAT = ['test']

def new_data():
    for category in CAT:

        path = os.path.join(DAT,category)

        for img in (os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            test_data.append([new_array, img])

new_data()
import pandas as pd

Xt = []
yt =[]

for features,lab in test_data:
    Xt.append(features)
    yt.append(lab)
    
Xt = np.array(Xt)
yt1 = pd.Series(yt).str.replace(".jpg","")
yt1 = pd.Series(yt).str.replace(".jpeg","")
Xt = np.array(Xt).reshape(-1, IMG_SIZE, IMG_SIZE)

Xt = Xt/255.0
pred = model.predict(Xt)

predict = pd.DataFrame(pred, index=yt1)
plt.plot(predict)
predict[0] = np.where(predict[0] <= 0.2, 0,predict[0])
predict[0] = np.where(predict[0] >= 0.2, 1,predict[0])
predict[0] = predict.astype(int)
predict.to_csv('C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem1/Attempt2Pizza.csv')
predict.index

