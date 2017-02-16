from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from sklearn.model_selection import train_test_split
import pickle


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


dataset_path      = "tiny-imagenet-200/train/"

BATCH_SIZE    = 256
EPOCHS        = 1000

with open('train.p', 'rb') as f:
    data = pickle.load(f)
print('features shape = {}'.format(data['features'].shape))
print('labels shape = {}'.format(data['labels'].shape))


features_all = data['features']
labels_all = data['labels']

CLASSES = len(np.unique(labels_all))
print("Total classes = {}".format(CLASSES))

features_all, labels_all = shuffle(features_all, labels_all)

a = -0.5
b = 0.5

min_f = np.min(features_all)
max_f = np.max(features_all)

features_all = a + (b-a) * (features_all - min_f) / (max_f - min_f)

X_train, y_train = features_all, labels_all

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

lb = LabelBinarizer()
y_train_one_hot = lb.fit_transform(y_train)

model = Sequential()

model.add(Convolution2D(64, 3, 3, input_shape=(64, 64, 3), border_mode = 'same'))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2,2)))

#model.add(Convolution2D(128, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(Convolution2D(128, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(MaxPooling2D((2,2)))

#model.add(Convolution2D(256, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(Convolution2D(256, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(Convolution2D(256, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(MaxPooling2D((2,2)))

#model.add(Convolution2D(512, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(Convolution2D(512, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(Convolution2D(512, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(MaxPooling2D((2,2)))

#model.add(Convolution2D(512, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(Convolution2D(512, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(Convolution2D(512, 3, 3, border_mode = 'same'))
#model.add(Activation('relu'))

#model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(200))    # Was 4096
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(200))    # Was 4096
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(CLASSES))
model.add(Activation('relu'))

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

history = model.fit(X_train, y_train_one_hot, nb_epoch=100, validation_split=0.2)

print('Validation accuracy of model = {}'.format(history['val_acc']))
