#########################################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
#############################################################
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Rescaling
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import image_dataset_from_directory, normalize
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC
from tensorflow import keras
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
####################################################
'''image_directory = 'home/sairaman/Desktop/stead_dataset/data/'

SIZE = 350
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

#parasitized_images = os.listdir(image_directory + 'Parasitized/')
noise_images = os.listdir(image_directory + 'NOISE')
for i, image_name in enumerate(noise_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'eq/' + image_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

earthquake_images = os.listdir(image_directory + 'EQ')
for i, image_name in enumerate(earthquake_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'noise/' + image_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)'''

#base_dir = "/home/sairaman/Desktop/stead-dataset/data/reshaped_images"
base_dir = "../data/reshaped_images"

train_ds = image_dataset_from_directory(base_dir,
                                        validation_split=0.2,
                                        subset="training",
                                        seed=123,
                                        image_size=(300, 300),
                                        batch_size=32)
val_ds = image_dataset_from_directory(base_dir,
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(300, 300),
                                      batch_size=32)
test = image_dataset_from_directory(base_dir, 
                                    seed = 123,
                                    image_size = (300, 300),
                                    batch_size = 32)
'''
train_ds = image_dataset_from_directory(base_dir / "train",
                                             image_size=(300, 300),
                                             batch_size=32)
val_ds = image_dataset_from_directory(base_dir / "validation",
                                             image_size=(300, 300),
                                             batch_size=32)                                                  

test_ds = image_dataset_from_directory(base_dir / "test",
                                             image_size=(300, 300),
                                             batch_size=32)
'''

'''# plotting some figures
class_names=train_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    print(labels[i])
    print(class_names,class_names[labels[i]])
    plt.title(class_names[labels[i]])
    plt.axis("off")
'''

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



INPUT_SHAPE = (300, 300, 3)   #change to (SIZE, SIZE, 3)

model = Sequential([
    Rescaling(1./255, input_shape=INPUT_SHAPE),
    Conv2D(16, 5, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
    Conv2D(32, 5, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
    Flatten(),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='sigmoid'),
    Dense(1)])

metrics = [TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'), FalseNegatives(name='fn'),
           BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
model.compile(loss='binary_crossentropy',
              optimizer='adam',            
              metrics=metrics)

print(model.summary())    
###############################################################  
checkpoint_filepath = "../data/model/checkpoint"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
epochs=1
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
model.save('../data/model/trial')
model = keras.models.load_model("../data/model/trial")
print(model.evaluate(test))

'''
from sklearn.metrics import classification_report
classes = train_ds.class_names
print(classification_report(labels_entire, pred_entire, target_names=classes))
'''
##############################################################
# Evaluation metrics
# Performance Graph

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
print(epochs_range)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig("../evolution_curve.png")

#Confusion Matrix and Classification Report
'''from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
Y_pred = model.predict(val_ds)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y, y_pred))
print('Classification Report')
target_names = ['EARTHQ','NOISE']
print(classification_report(y, y_pred, target_names=target_names))
cm = confusion_matrix(y,y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()'''