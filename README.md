TextCNN解读
===========

# 一. 简介

> TextCNN 是利用卷积神经网络对文本进行分类的算法，由 Yoon Kim 在2014年的《Convolutional Neural Networks for Sentence Classification》一文中提出。

# 二. 结构

> 论文中TextCNN结构如图所示：

img1

> 网络上更详细的图如下：

img2

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

> 在TextCNN中用的是一维的max pooling，当然也可以使用(dynamic) k-max pooling，在pooling阶段保留 k 个最大值，保留全局信息。**TextCNN模型最大的问题也是这个全局的max pooling丢失了结构信息，因此很难去发现文本中的转折关系等复杂模式，**TextCNN只能知道哪些关键词是否在文本中出现了，以及相似度强度分布，而不可能知道哪些关键词出现了几次以及出现这些关键词出现顺序。假想一下如果把这个中间结果给人来判断，人类也很难得到对于复杂文本的分类结果，所以机器显然也做不到。**针对这个问题，可以尝试k-max pooling做一些优化，k-max pooling针对每个卷积核都不只保留最大的值**，它保留前k个最大值，并且保留这些值出现的顺序，也即按照文本中的位置顺序来排列这k个最大值，在某些比较复杂的文本上相对于1-max pooling会有提升。

# 四. 代码

> 官方在github上公开了一个使用实例，具体见：https://github.com/dennybritz/cnn-text-classification-tf.，具体解读见：https://hunto.github.io/nlp/2018/03/29/TextCNN%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E8%AF%A6%E8%A7%A3.html. 和 http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/.



