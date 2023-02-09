import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
base_dir = "../data/reshaped_images"
test_ds = image_dataset_from_directory(base_dir,
                                      validation_split=0.05,
                                      subset="validation",
                                      seed=123,
                                      image_size=(300, 300),
                                      batch_size=1)
print(test_ds.class_names)   
model = tf.keras.models.load_model("../data/model/7-02_20epochs")  
#print(model.evaluate(test))                       
<<<<<<< HEAD
for image_batch,label_batch in test.as_numpy_iterator():
    print(model.predict_on_batch(image_batch))
    #################
=======
label = []
inp = []
for x,y in test_ds.as_numpy_iterator():
    label.append(y[0]))
    inp.append(x)
    #print(y)
print(inp.shape)
>>>>>>> aa826f8b19a84af1e1703b6e536a2f539cd064fa
