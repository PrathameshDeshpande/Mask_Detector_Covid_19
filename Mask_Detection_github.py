import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
import cv2
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from keras import regularizers
print(os.getcwd)

print(len(os.listdir('with_mask')))
print(len(os.listdir('without_mask')))

try:
    os.mkdir('C:/Users/91797/Desktop/Mask-CNN/withm-withoutm')
    os.mkdir('C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/training')
    os.mkdir('C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/testing')
    os.mkdir('C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/training/withm')
    os.mkdir('C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/training/withoutm')
    os.mkdir('C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/testing/withm')
    os.mkdir('C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/testing/withoutm')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

    dataset = []
    
    for unitData in os.listdir(SOURCE):
        data = SOURCE + '/' + unitData
        if(os.path.getsize(data) > 0):
            dataset.append(unitData)
        else:
            print('Skipped ' + unitData)
            print('Invalid file i.e zero size')
    
    train_set_length = int(len(dataset) * SPLIT_SIZE)
    test_set_length = int(len(dataset) - train_set_length)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = dataset[0:train_set_length]
    test_set = dataset[-test_set_length:]
       
    for unitData in train_set:
        temp_train_set = SOURCE + "/" + unitData
        final_train_set = TRAINING + "/" + unitData
        copyfile(temp_train_set, final_train_set)
    
    for unitData in test_set:
        temp_test_set = SOURCE + '/' + unitData
        final_test_set = TESTING + '/' + unitData
        copyfile(temp_test_set, final_test_set)
        
with_mask_dir = 'C:/Users/91797/Desktop/Mask-CNN/with_mask'
training_with_mask_dir = 'C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/training/withm'
testing_with_mask_dir =  'C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/testing/withm'
without_mask_dir = 'C:/Users/91797/Desktop/Mask-CNN/without_mask'
training_without_mask_dir = 'C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/training/withoutm'
testing_without_mask_dir = 'C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/testing/withoutm'

split_size = .8
split_data(with_mask_dir,training_with_mask_dir,testing_with_mask_dir,split_size)
split_data(without_mask_dir,training_without_mask_dir,testing_without_mask_dir,split_size)
   

print(len(os.listdir(training_with_mask_dir)))
print(len(os.listdir(testing_with_mask_dir)))
print(len(os.listdir(training_without_mask_dir)))
print(len(os.listdir(testing_without_mask_dir)))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50,activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


Training_dir = 'C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/training'
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(Training_dir, 
                                                    batch_size=10, 
                                                    class_mode='categorical', 
                                                    target_size=(150,150))

Validation_dir = 'C:/Users/91797/Desktop/Mask-CNN/withm-withoutm/testing'
validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(Validation_dir,batch_size=10,class_mode='categorical',target_size=(150,150))

history = model.fit_generator(train_generator,
                              epochs=20,
                              verbose=1,
                              validation_data=validation_generator)


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.title('Training and validation loss')
model.save('mask_trained2.h5') 

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(1) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()


  