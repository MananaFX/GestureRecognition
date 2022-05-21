import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.framework import ops
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Activation
import DNN

'''
对测试集与训练集的预处理
'''
train_dataset = h5py.File('datasets/train_signs.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File('datasets/test_signs.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])
test_set_y_orig = np.array(test_dataset["test_set_y"][:])

print(train_set_x_orig.shape)
print(test_set_x_orig.shape)

x_train = train_set_x_orig / 255.0
x_test = test_set_x_orig / 255.0

y_train = keras.utils.to_categorical(train_set_y_orig, 6)  ##独热编码，对应0-6的输出
y_test = keras.utils.to_categorical(test_set_y_orig, 6)

'''
图像转换为灰度图
'''


def RgbtoGray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


x_train_gray = []
for i in x_train:
    x_train_gray.append(RgbtoGray(i))

x_test_gray = []
for i in x_test:
    x_test_gray.append(RgbtoGray(i))

x_train_gray = np.array(x_train_gray)
x_test_gray = np.array(x_test_gray)


# print(x_train[1:].shape)


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=0)
    return a


'''
DNN模型搭建
'''


def model(x_train, Y_train, x_test, Y_test,
          learning_rate=0.0001, num_epochs=1000, minibatch_size=32,
          print_cost=True, is_plot=True):
    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.random.set_seed(1)
    seed = 3
    X_train = x_train.reshape(x_train.shape[0], -1).T  # 进行flatten处理，X
    X_test = x_test.reshape(x_test.shape[0], -1).T
    Y_train = Y_train.T
    Y_test = Y_test.T
    (n_x, m) = X_train.shape  # 获取输入节点数量和样本数
    n_y = Y_train.shape[0]  # 获取输出节点数量
    costs = []  # 成本集

    # 给X和Y创建placeholder
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.compat.v1.placeholder(tf.float32, [n_y, None], name="Y")

    parameters = DNN.initialize_parameters()  ##对参数进行初始化

    Z3 = DNN.forward_propagation(X, parameters)  # 前向传播

    cost = DNN.compute_cost(Z3, Y)  # 计算成本

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # 反向传播，使用Adam优化

    init = tf.compat.v1.global_variables_initializer()  # 初始化所有的变量

    with tf.compat.v1.Session() as sess:

        sess.run(init)  # 初始化

        for epoch in range(num_epochs):

            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)  # minibatch的总数量，即每次喂给网络的样本数
            seed = seed + 1
            minibatches = DNN.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # 选择一个minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # 数据已经准备好了，开始运行session
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches

            # 每二十个打印一次cost
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                # 是否打印：
                if print_cost and epoch % 20 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))        # 计算结果
        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("验证集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


'''
CNN模型 modelC 的构造
'''
modelC = Sequential()
modelC.call = tf.function(modelC.call)
modelC.add(Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding='same',
                  input_shape=(64, 64, 3)))  ##卷积核设为4*4,输入3维，输出8维
modelC.add(BatchNormalization())
modelC.add(Activation('relu'))
modelC.add(Dropout(0.1))  ##用于防止过拟合
modelC.add(MaxPooling2D(pool_size=(8, 8), strides=(8, 8)))  # 添加最大池化层
modelC.add(Activation('relu'))
modelC.add(Conv2D(16, (2, 2), strides=(1, 1), padding='same'))  ##卷积核为2，输出16维
modelC.add(BatchNormalization())
modelC.add(Activation('relu'))
modelC.add(Dropout(0.2))
modelC.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

modelC.add(Flatten())  # 将上一层输出的数据变成一维

modelC.add(Dense(128, activation='relu'))  # 添加全连接层 //给输出使用激活函数
modelC.add(Dense(64, activation='relu'))
# modelC.add(Dropout(0.5))
modelC.add(Dense(6, activation='softmax'))  # 对应6个数字的输出

# print(modelC.summary())

modelC.compile(loss='categorical_crossentropy',
               optimizer='adagrad',
               metrics=['accuracy']
               )  # 模型编译

'''
DNN模型 modelD 的构造

modelD = Sequential()
modelD.call = tf.function(modelD.call)
modelD.add(Flatten(input_shape=(64, 64)))
modelD.add(Dense(256, activation='relu'))
modelD.add(Dense(128, activation='relu'))
modelD.add(Dropout(0.1))
modelD.add(Dense(64, activation='relu'))
modelD.add(Dropout(0.2))
modelD.add(Dense(32, activation='relu'))
modelD.add(Dropout(0.4))
modelD.add(Dense(6, activation='softmax'))

# print(modelD.summary())

modelD.compile(loss='categorical_crossentropy',
               optimizer='adagrad',
               metrics=['accuracy']
               )  # 模型编译
'''

'''
模型的训练以及测试
'''

modelC.fit(x_train, y_train, epochs=50, batch_size=64)  # 模型训练
# modelD.fit(x_train_gray, y_train, epochs=800, batch_size=64)
test_lossC, test_accC = modelC.evaluate(x_test, y_test,
                                        verbose=2, batch_size=64)  # 对测试集的评估
print('\nCNN验证集的准确率:\t', test_accC)
predC = modelC.predict(x_test)  # pred数组存放对测试集的预测值，用于接下来图表的输出

parameters = model(x_train, y_train, x_test, y_test, num_epochs=100)
predD = np.empty(shape=(120, 6))
X_test = x_test.reshape(120, 1, 64 * 64 * 3)
print(X_test.shape)
for i in range(X_test.shape[0]):
    predD[i] = np.squeeze(DNN.predict(X_test[i].T, parameters))
    predD[i] = soft_max(predD[i])
print(predD.shape)
print(predD[1])
print(predD[2])
'''
结果的展示 
'''


# test_lossD, test_accD = modelD.evaluate(x_test_gray, y_test,verbose=2, batch_size=64)  # 对测试集的评估
# print('\nDNN Test accuracy:', test_accD)

# predD = modelD.predict(x_test_gray)


def plot_image(i, predictions_array, true_label, modelName):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} Pred: {} Conf: {:2.0f}% (True: {})".format(modelName,
                                                              predicted_label,
                                                              100 * np.max(predictions_array),
                                                              test_set_y_orig[i]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(6))
    plt.yticks([])
    thisplot = plt.bar(range(6), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 8
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(x_test[i])
plt.subplot(1, 3, 2)
plot_image(i, predC[i], test_set_y_orig, 'CNN')
plot_value_array(i, predC[i], test_set_y_orig)
plt.subplot(1, 3, 3)
plot_image(i, predD[i], test_set_y_orig, 'DNN')
plot_value_array(i, predD[i], test_set_y_orig)

plt.show()
