#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


# In[3]:


data_url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"2


# In[4]:


data_dir=tf.keras.utils.get_file('flower_photos',origin=data_url,untar=True)


# In[5]:


data_dir=pathlib.Path(data_dir)


# In[6]:


image_count=len(list(data_dir.glob('*/*.jpg')))


# In[7]:


print(image_count)


# In[8]:


print(os.listdir(data_dir))


# In[9]:


roses=list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))


# In[10]:


tulips=list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[5]))


# In[11]:


#defining image structure
batch_size=32
height=180
width=180


# In[12]:


train=tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,subset="training",seed=123,image_size=(height,width),
                                                 batch_size=batch_size)


# In[13]:


test=tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,subset="validation",seed=123,image_size=(height,width),
                                                 batch_size=batch_size)
                                    


# In[14]:


class_names=train.class_names
print(class_names)


# In[15]:


plt.figure(figsize=(12,12))
for images,labels in train.take(1):
    for i in range(12):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    
    
    


# In[16]:


num_classes=len(class_names)
model=Sequential([
    layers.experimental.preprocessing.Rescaling(1./255,input_shape=(height,width,3)),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(num_classes,activation='softmax')
    
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
model.summary()






# In[17]:


epochs=15
history=model.fit(train,validation_data=test,epochs=epochs)


# In[ ]:


def predict_input_image(img):
    img_4d=img.reshape(-1,180,180,3)
    prediction=model.predict(img_4d)[0]
    return{class_names[i]:float(prediction[i]) for i in range(5)}
import gradio as gr
image=gr.inputs.Image(shape=(180,180))
label=gr.outputs.Label(num_top_classes=5)
gr.Interface(fn=predict_input_image,inputs=image,outputs=label,interpretation='default').launch(debug='True')

