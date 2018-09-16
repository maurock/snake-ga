import keras
from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 500)

traintest=pd.read_csv('snakemove6.csv')
x=traintest.drop(['3.1'], axis=1)
#x=x.iloc[1:,:]
y=traintest['3.1']

def network(weights_path=None):
    model = Sequential()
    model.add(Dense(21,input_dim = 21,activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def fitness(x_food, x, y_food, y, food):
    score = (1-(np.sqrt((x_food-x)**2 + (y_food - y)**2))/566 + food)
    return score

model = network()


x=x.values
y=y-1
y_onehot = keras.utils.to_categorical(y, num_classes=4)
#print(y_onehot[:])
lr_init = 0.0001
optimizer = keras.optimizers.Adam(lr= lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss= 'sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x,y, epochs=20 ,validation_split=0.2)
model.save('snake8.h5')

#snake5.h5 = 408 variables
#snake6.h5 = 8 variables (big architecture)
#snake7.h5 = 8 variables (small architecture)
#snake8.h5 = 17 variables (small architecture)
