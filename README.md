TextCNN解读
===========

# 一. 简介

> TextCNN 是利用卷积神经网络对文本进行分类的算法，由 Yoon Kim 在2014年的《Convolutional Neural Networks for Sentence Classification》一文中提出。

# 二. 结构

> 论文中TextCNN结构如图所示：

![image](https://github.com/ShaoQiBNU/TextCNN/blob/master/image/1.png)

> 网络上更详细的图如下：

![image](https://github.com/ShaoQiBNU/TextCNN/blob/master/image/2.png)

> 网络结构详解如下：

## 1. 第一层

> 第一层输入的是 7 x 5 的词向量矩阵，词向量的维度为5，共7个单词。词向量可由word2vec、fasttext等生成，具体可见： http://frankchen.xyz/2017/12/18/How-to-Use-Word-Embedding-Layers-for-Deep-Learning-with-Keras/.

## 2. 第二层

> 第二层是卷积层，共有6个卷积核。
>
> 卷积核的宽度为词向量的维度即5，卷积核的高度可以自己定义，此处设置为4、3和2，卷积的步长为1。6个卷积核的尺寸为 4 x 5、3 x 5和2 x 5，每个尺寸各两个。
>
> 输入层分别与6个卷积核进行卷积操作，再利用激活函数激活，每个卷积核都得到对应的feature maps。feature map的大小为：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{c}&space;=&space;[c_1,&space;c_2,&space;...&space;,&space;c_{n-h&plus;1}]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\mathbf{c}&space;=&space;[c_1,&space;c_2,&space;...&space;,&space;c_{n-h&plus;1}]" title="\mathbf{c} = [c_1, c_2, ... , c_{n-h+1}]" /></a>
>
> 其中，<a href="https://www.codecogs.com/eqnedit.php?latex=n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n" title="n" /></a>为单词个数即7，<a href="https://www.codecogs.com/eqnedit.php?latex=h" target="_blank"><img src="https://latex.codecogs.com/svg.latex?h" title="h" /></a>为卷积核高度即4、3和2。

## 3. 第三层

> 第三层是池化层，采用max-pooling的方式提取出每个feature map的最大值，然后进行级联，得到 6 x 1 的特征。

## 4. 第四层

> 第四层是全连接的softmax输出层，该层可进行正则化操作。

# 三. 细节

### 1. 词向量

> 词向量有静态和非静态的，静态的可以使用pre-train的，非静态的则可以在训练过程中进行更新。一般推荐非静态的fine-tunning方式，即以pre-train的词向量进行初始化，然后在训练过程中进行调整，它能加速收敛。

## 2. channel

> 图像中可以利用 (R, G, B) 作为不同channel，而文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。

### 3. conv-1d

> 在TextCNN中用的是一维卷积（conv-1d），一维卷积带来的问题是需要设计通过不同size的filter获取不同宽度的视野。

### 4. max pooling

> 在TextCNN中用的是一维的max pooling，当然也可以使用(dynamic) k-max pooling，在pooling阶段保留 k 个最大值，保留全局信息。TextCNN模型最大的问题也是这个全局的max pooling丢失了结构信息，因此很难去发现文本中的转折关系等复杂模式，TextCNN只能知道哪些关键词是否在文本中出现了，以及相似度强度分布，而不可能知道哪些关键词出现了几次以及出现这些关键词出现顺序。假想一下如果把这个中间结果给人来判断，人类也很难得到对于复杂文本的分类结果，所以机器显然也做不到。针对这个问题，可以尝试k-max pooling做一些优化，k-max pooling针对每个卷积核都不只保留最大的值，它保留前k个最大值，并且保留这些值出现的顺序，也即按照文本中的位置顺序来排列这k个最大值，在某些比较复杂的文本上相对于1-max pooling会有提升。

# 四. 代码

> 利用Keras框架实现文本分类，代码及结果如下：

```python
###################### load packages ####################
from keras.datasets import imdb
from keras import preprocessing
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Convolution1D, Flatten, Dropout, MaxPool1D
from keras.utils.np_utils import to_categorical


###################### load data ####################
######### 只考虑最常见的1000个词 ########
num_words = 1000

######### 导入数据 #########
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

print(x_train.shape)
print(x_train[0][:5])

print(y_train.shape)
print(y_train[0])


###################### preprocess data ####################
######## 句子长度最长设置为20 ########
max_len = 20

######## 对文本进行填充，将文本转成相同长度 ########
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

print(x_train.shape)
print(x_train[0])

######## 对label做one-hot处理 ########
num_class = 2
y_train = to_categorical(y_train, num_class)
y_test = to_categorical(y_test, num_class)

print(y_train.shape)
print(y_train[0])


###################### build network ####################
######## word dim 词向量维度 ########
word_dim = 300

######## network structure ########
#### input层 ####
main_input = Input(shape=(max_len,))

#### Embedding层 ####
embedding = Embedding(input_dim=1000, output_dim=word_dim, input_length=max_len)(main_input)

#### 卷积层1 ####
conv1 = Convolution1D(256, 3, padding='valid', strides=1, activation='relu')(embedding)
pool1 = MaxPool1D(pool_size=(max_len-3+1))(conv1)

#### 卷积层2 ####
conv2 = Convolution1D(256, 4, padding='valid', strides=1, activation='relu')(embedding)
pool2 = MaxPool1D(pool_size=(max_len-4+1))(conv2)

#### 卷积层3 ####
conv3 = Convolution1D(256, 5, padding='valid', strides=1, activation='relu')(embedding)
pool3 = MaxPool1D(pool_size=(max_len-5+1))(conv3)

#### 级联层 ####
cnn = concatenate([pool1, pool2, pool3], axis=-1)

#### flatten ####
flat = Flatten()(cnn)

#### drop ####
drop = Dropout(0.2)(flat)

#### 输出层 ####
out = Dense(num_class, activation='softmax')(drop)

#### 构建模型 ####
model = Model(inputs=main_input, outputs=out)
print(model.summary())

######## optimization and train ########
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_data=(x_test, y_test))
```

```
(25000,)
[1, 14, 22, 16, 43]
(25000,)
1
(25000, 20)
[ 65  16  38   2  88  12  16 283   5  16   2 113 103  32  15  16   2  19
 178  32]
(25000, 2)
[0. 1.]
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_9 (InputLayer)            (None, 20)           0                                            
__________________________________________________________________________________________________
embedding_32 (Embedding)        (None, 20, 300)      300000      input_9[0][0]                    
__________________________________________________________________________________________________
conv1d_21 (Conv1D)              (None, 18, 256)      230656      embedding_32[0][0]               
__________________________________________________________________________________________________
conv1d_22 (Conv1D)              (None, 17, 256)      307456      embedding_32[0][0]               
__________________________________________________________________________________________________
conv1d_23 (Conv1D)              (None, 16, 256)      384256      embedding_32[0][0]               
__________________________________________________________________________________________________
max_pooling1d_21 (MaxPooling1D) (None, 1, 256)       0           conv1d_21[0][0]                  
__________________________________________________________________________________________________
max_pooling1d_22 (MaxPooling1D) (None, 1, 256)       0           conv1d_22[0][0]                  
__________________________________________________________________________________________________
max_pooling1d_23 (MaxPooling1D) (None, 1, 256)       0           conv1d_23[0][0]                  
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 1, 768)       0           max_pooling1d_21[0][0]           
                                                                 max_pooling1d_22[0][0]           
                                                                 max_pooling1d_23[0][0]           
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 768)          0           concatenate_7[0][0]              
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 768)          0           flatten_4[0][0]                  
__________________________________________________________________________________________________
dense_26 (Dense)                (None, 2)            1538        dropout_4[0][0]                  
==================================================================================================
Total params: 1,223,906
Trainable params: 1,223,906
Non-trainable params: 0
__________________________________________________________________________________________________
None
Train on 25000 samples, validate on 25000 samples
Epoch 1/20
25000/25000 [==============================] - 15s 609us/step - loss: 0.6072 - acc: 0.6584 - val_loss: 0.5224 - val_acc: 0.7336
Epoch 2/20
25000/25000 [==============================] - 11s 446us/step - loss: 0.4877 - acc: 0.7590 - val_loss: 0.5032 - val_acc: 0.7464
Epoch 3/20
25000/25000 [==============================] - 11s 454us/step - loss: 0.4411 - acc: 0.7926 - val_loss: 0.4954 - val_acc: 0.7489
Epoch 4/20
25000/25000 [==============================] - 12s 479us/step - loss: 0.3923 - acc: 0.8220 - val_loss: 0.4973 - val_acc: 0.7517
Epoch 5/20
25000/25000 [==============================] - 11s 453us/step - loss: 0.3333 - acc: 0.8664 - val_loss: 0.4999 - val_acc: 0.7537
Epoch 6/20
25000/25000 [==============================] - 12s 472us/step - loss: 0.2650 - acc: 0.9083 - val_loss: 0.5201 - val_acc: 0.7479
Epoch 7/20
25000/25000 [==============================] - 11s 450us/step - loss: 0.1974 - acc: 0.9407 - val_loss: 0.5360 - val_acc: 0.7496
Epoch 8/20
25000/25000 [==============================] - 11s 450us/step - loss: 0.1389 - acc: 0.9691 - val_loss: 0.5784 - val_acc: 0.7499
Epoch 9/20
25000/25000 [==============================] - 12s 462us/step - loss: 0.0930 - acc: 0.9846 - val_loss: 0.6162 - val_acc: 0.7472
Epoch 10/20
25000/25000 [==============================] - 12s 474us/step - loss: 0.0649 - acc: 0.9920 - val_loss: 0.6538 - val_acc: 0.7482
Epoch 11/20
25000/25000 [==============================] - 11s 454us/step - loss: 0.0455 - acc: 0.9962 - val_loss: 0.7021 - val_acc: 0.7465
Epoch 12/20
25000/25000 [==============================] - 11s 443us/step - loss: 0.0337 - acc: 0.9971 - val_loss: 0.7383 - val_acc: 0.7470
Epoch 13/20
25000/25000 [==============================] - 12s 469us/step - loss: 0.0247 - acc: 0.9989 - val_loss: 0.7794 - val_acc: 0.7457
Epoch 14/20
25000/25000 [==============================] - 11s 447us/step - loss: 0.0191 - acc: 0.9990 - val_loss: 0.8106 - val_acc: 0.7464
Epoch 15/20
25000/25000 [==============================] - 11s 456us/step - loss: 0.0152 - acc: 0.9993 - val_loss: 0.8336 - val_acc: 0.7466
Epoch 16/20
25000/25000 [==============================] - 11s 442us/step - loss: 0.0123 - acc: 0.9997 - val_loss: 0.8612 - val_acc: 0.7462
Epoch 17/20
25000/25000 [==============================] - 11s 441us/step - loss: 0.0105 - acc: 0.9996 - val_loss: 0.8866 - val_acc: 0.7450
Epoch 18/20
25000/25000 [==============================] - 12s 465us/step - loss: 0.0084 - acc: 0.9997 - val_loss: 0.9129 - val_acc: 0.7452
Epoch 19/20
25000/25000 [==============================] - 11s 447us/step - loss: 0.0072 - acc: 0.9998 - val_loss: 0.9384 - val_acc: 0.7456
Epoch 20/20
25000/25000 [==============================] - 12s 460us/step - loss: 0.0062 - acc: 0.9998 - val_loss: 0.9500 - val_acc: 0.7468
```

> 官方在github上公开了一个tensorflow使用实例，具体见：https://github.com/dennybritz/cnn-text-classification-tf. 解读见：https://hunto.github.io/nlp/2018/03/29/TextCNN%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E8%AF%A6%E8%A7%A3.html. 和 http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/.



