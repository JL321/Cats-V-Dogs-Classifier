#CNN Classifier based on Dog vs Cat database on Kaggle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()

TRAIN_DIR = r'C:\Users\david\Desktop\CatsVDogs\train' #Set your own directory (location of files)
TEST_DIR = r'C:\Users\david\Desktop\CatsVDogs\test1'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats--{}--{}.model'.format(LR, '2conv-basic') #Save Name

def label_img(img): #Label Images 
    word_label= img.split('.')[-3]
    if word_label == 'cat': 
        return [1,0]
    elif word_label == 'dog':
        return [0,1]
    
def create_train_data():
    train_data = [] #tqdm is a smart progress bar
    for img in tqdm(os.listdir(TRAIN_DIR)): #listdir for file quantity
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE)) #features- grayscale for speed
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data

def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)): #listdir for file quantity
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE)) #features
        test_data.append([np.array(img), img_num])
    np.save('test_data.npy', test_data)
    return test_data    

#train_data = create_train_data()
train_data = np.load('train_data.npy')

net = input_data(shape=[None, IMG_SIZE, IMG_SIZE,1], name = 'input')

net = conv_2d(net, 32, 3, activation = 'relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 3, activation = 'relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 3, activation = 'relu')
net = max_pool_2d(net, 2, strides = 2)
net = conv_2d(net, 128, 5, activation = 'relu')
net = max_pool_2d(net, 2, strides = 2)
net = conv_2d(net, 128, 5, activation = 'relu')
net = max_pool_2d(net, 2, strides = 2)
net = conv_2d(net, 256, 3, activation = 'relu')
net = conv_2d(net, 384, 3, activation = 'relu')
net = conv_2d(net, 512, 3, activation = 'relu')
net = conv_2d(net, 256, 3, activation = 'relu')
net = max_pool_2d(net, 2, strides = 2)

net = flatten(net)
net = fully_connected(net, 1024, activation = 'relu')
net = dropout(net, .8)
net = fully_connected(net, 256, activation = 'relu')
net = fully_connected(net, 2, activation = 'softmax')

net = regression(net, optimizer='adam', learning_rate = LR,loss ='categorical_crossentropy', name = 'targets')

model = tflearn.DNN(net, tensorboard_dir = 'log')

#if os.path.exists('{}.meta'.format(MODEL_NAME)): -Don't start from nothing!
#    model.load(MODEL_NAME)
#    print("loaded!")
    
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input':X},{'targets':Y}, n_epoch = 1, validation_set = ({'input':test_x},{'targets':test_y}),
          show_metric = True, snapshot_step = 500, run_id = MODEL_NAME)

#tensorboard --logdir=foo:C:\Users\david\Desktop\CatsVDogs\log- load tensorboard at your log directory

model.save(MODEL_NAME)

#test_data = process_test_data()
test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1:
        str_label='Dog'
    elif np.argmax(model_out) == 0:
        str_label ='Cat'
        
    y.imshow(orig, cmap = 'gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
    