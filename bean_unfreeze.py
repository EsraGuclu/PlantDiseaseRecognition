from google.colab import drive
drive.mount('/content/drive')

import numpy as np 
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import applications  
import matplotlib.pyplot as plt

img_width, img_height = 224, 224  
epochs = 50  
batch_size = 32 

#datasets directories 
train_data_dir = '/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/train'  
validation_data_dir = '/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/validation'

#build network without last layer/top layer
#model = applications.VGG16(include_top=False, weights='imagenet') 
#model = applications.MobileNetV2(include_top=False, weights='imagenet') 
model = applications.ResNet50V2(include_top=False, weights='imagenet')

output = model.output
flatten = GlobalAveragePooling2D()(output)
fc1 = Dense(256, activation='relu')(flatten)
fc2 = Dropout(0.5)(fc1)
fc3 = Dense(128, activation='relu')(fc2)
fc4 = Dropout(0.2)(fc3)
fc5 = Dense(64, activation='relu')(fc4)
fc6 = Dropout(0.2)(fc5)
pred = Dense(3, activation='softmax')(fc6)
new_model = Model(model.input, pred)



new_model.summary()

vgg_layer = 15          #19
mobilenet_layer = 140   #154
resnet_layer = 170      #190

for layer in model.layers[:resnet_layer]:
    layer.trainable = False


new_model.compile(optimizer='adam',  
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

# Add data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
history = new_model.fit(train_generator, 
                        epochs=epochs,  
                        batch_size=batch_size,  
                        validation_data=validation_generator,
                        verbose = 1,
                        callbacks=[callbacks])  


(loss, acc) = new_model.evaluate(validation_generator, 
                                  batch_size=batch_size, 
                                  verbose=1)

print("[INFO] accuracy: {:.2f}%".format(acc * 100))  
print("[INFO] Loss: {}".format(loss))

plt.figure(1)  

# summarize for accuracy  
plt.subplot(211)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  
plt.show() 

# summarize for loss  
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  
plt.show()

test_dir = '/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/test'

datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = datagen.flow_from_directory( test_dir,  
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,  
                                              class_mode=None,  
                                              shuffle=False)  

test_generator.reset()
   
pred= new_model.predict_generator(test_generator)
predicted_class_indices=np.argmax(pred, axis =1 )
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predicted_class_indices)
print(labels)
print(predictions)

