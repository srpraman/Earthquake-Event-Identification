'''import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

train_dataset = tf.convert_to_tensor(np.load("../data/waveforms/chunk2_to_1d_array_train_normalized.npz")['arr_0'], dtype=tf.float32)
test_dataset = tf.convert_to_tensor(np.load("../data/waveforms/chunk2_to_1d_array_test_normalized.npz")['arr_0'], dtype=tf.float32)
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

input_layer = Input(shape = (3500, ))
# Building the Encoder network
encoded = Dense(64, activation ='tanh')(input_layer)
encoded = Dense(32, activation ='tanh')(encoded)
encoded = Dense(16, activation ='tanh')(encoded)
# Building the Decoder network
decoded = Dense(32, activation ='tanh')(encoded)
decoded = Dense(64, activation ='tanh')(decoded)
  
# Building the Output Layer
output_layer = Dense(3500, activation ='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer ="adam", loss ="mse")
print(autoencoder.summary()) 
# Training the Auto-encoder network
autoencoder.fit(test_dataset, test_dataset, epochs = 10, batch_size=256, shuffle = True, validation_split = 0.20)'''

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Rescaling
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import image_dataset_from_directory, normalize, get_file
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC
from keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow import keras
import h5py
import numpy as np
import time

test = np.load("../data/waveforms/chunk2_to_1d_array_train_normalized.npz")
data = test["arr_0"].reshape((86376,3500,1))
# define model

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(3500,1)))
model.add(RepeatVector(3500))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
model.fit(data, data, epochs=30, verbose=1, batch_size = 32)


'''
import matplotlib.pyplot as plt
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Rescaling
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import image_dataset_from_directory, normalize
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC
from tensorflow import keras
from keras import layers, Model
import h5py

base_dir = "../data/reshaped_images"

train_ds = image_dataset_from_directory(base_dir,
                                      label_mode = None,
                                        validation_split=0.2,
                                        subset="training",
                                        seed=123,
                                        image_size=(300, 300),
                                        batch_size=32)
val_ds = image_dataset_from_directory(base_dir,
                                      
                                      label_mode = None,
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(300, 300),
                                      batch_size=32)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
model = Sequential([
    Rescaling(1./255, input_shape=INPUT_SHAPE),
    Conv2D(8, 5, padding='same', activation='tanh'),
    MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
    Conv2D(16, 5, padding='same', activation='tanh'),
    MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
    Flatten(),
    Dense(32, activation='tanh'),
    Dropout(0.2),
    Dense(16, activation='sigmoid'),
    Dense(1)])
input = layers.Input(shape=(300, 300, 3))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="tanh", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="tanh", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="tanh", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="tanh", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()
print(autoencoder.summary()) 
# Training the Auto-encoder network
autoencoder.fit(train_ds,
    train_ds,
    epochs=50,
    shuffle=True,validation_data=(val_ds,val_ds))'''