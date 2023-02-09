import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
base_dir = "../data/reshaped_images"

test =image_dataset_from_directory(base_dir,
                                      validation_split=0.1,
                                      subset="validation",
                                      seed=123,
                                      image_size=(300, 300),
                                      batch_size=32)
print(test.class_names)   
model = tf.keras.models.load_model("../data/model/7-02_20epochs")  
#print(model.evaluate(test))                       
for image_batch,label_batch in test.as_numpy_iterator():
    print(model.predict_on_batch(image_batch))