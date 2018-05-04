import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from os import walk
import warnings
from os.path import join
import numpy as np
import keras
from skimage import io, transform
from tflearn.layers.conv import global_avg_pool
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model
from keras import layers
from keras import backend as K
from keras.applications.vgg16 import preprocess_input
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)

KTF.set_session(session)
'''
# In[ ]:
x_train = []
y_train = []
x_test = []
y_test = []
for root, dirs, files in walk(r"CroppedYale"):
    data = []
    label = []
    for f in files:
        if f[-3:] != "pgm":
            continue
        img = io.imread(join(root, f))
        img = transform.resize(img, (224, 224), mode="constant")
        img = np.array(img, dtype=float)
        img = np.dstack((img, img, img))
        data.append(img)
        label.append(int(f[5:7]) - 1)
    x_test.extend(data[35:])
    x_train.extend(data[:35])
    y_test.extend(label[35:])
    y_train.extend(label[:35])

x_test = preprocess_input(np.array(x_test))
x_train = preprocess_input(np.array(x_train))
y_test = keras.utils.to_categorical(y_train, num_classes=39)
y_train = keras.utils.to_categorical(y_train, num_classes=39)

def vgg16(input_tensor=None, input_shape=None):

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fca')(x)
    # x = Dense(4096, activation='relu', name='fcb')(x)
    x = Dense(39, activation='softmax', name='Classification')(x)

    # Create model
    return Model(img_input, x, name='vgg16')

# In[ ]:


model = vgg16(input_shape=[224, 224, 3])
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
# model.load_weights("model.h5", by_name=True)
model.fit(x_train, y_train, batch_size=8, epochs=20)
score = model.evaluate(x_test, y_test)
print("\nLoss:", score[0])
print("Accuracy:", score[1])
