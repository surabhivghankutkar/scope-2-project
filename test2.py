import numpy as np 
import pandas as pd 

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

inputs=pd.read_csv(r'''C:\Users\Rohan\Downloads\New_training_dataset.csv''')
labels=inputs.iloc[:,0].values.astype('int32')
trainip=(inputs.iloc[:,1:].values).astype('float32')
testip=(pd.read_csv(r'''C:\Users\Rohan\Downloads\New_testing_dataset.csv''').values).astype('float32')

trainop=np_utils.to_categorical(labels)

scale=np.max(trainip)
trainip/=scale
testip/=scale

input_dim=trainip.shape[1]
nb_classes=trainop.shape[1]

model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(trainip,trainop,epochs=15,batch_size=16,validation_split=0.2,verbose=2)

preds=model.predict_classes(testip, verbose=1)

pd.DataFrame({"ImageId": list(range(1,len(preds)+1)),"Label": preds}).to_csv('testing_output1.csv', index=False, header=True)
