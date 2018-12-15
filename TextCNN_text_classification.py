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