import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def full_connection():
    # 用全连接对手写数字进行识别
    # 1、准备数据
    mnist = input_data.read_data_sets("../data/mnist_data", one_hot=True)
    #  用占位符定义真实数据
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784])  # 特征值
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])  # 真实值

    # 2、构建模型 - 全连接
    # y_predict[None,10]=X[None,784]*weights[784,10]+bias[10]
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10], stddev=0.01))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10], stddev=0.1))
    y_predict = tf.matmul(X, weights) + bias

    # 3、构造损失函数
    loss_list = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_true)
    loss = tf.reduce_mean(loss_list)

    # 4、优化损失(梯度下降)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 5、计算准确率 tf.argmax()计算最大值所在列数
    bool_list = tf.equal(tf.argmax(y_predict, axis=1), tf.argmax(y_true, axis=1))
    accuracy = tf.reduce_mean(tf.cast(bool_list, tf.float32))
    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 开始训练
        for i in range(100):
            # 获取真实值
            image, label = mnist.train.next_batch(100)
            # 因为optimizer返回的是None 所以用_,来接
            _, loss_value, accuracy_value = sess.run([optimizer, loss, accuracy], feed_dict={X: image, y_true: label})

            print("第%d次的损失为%f，准确率为%f" % (i + 1, loss_value, accuracy_value))

    return None


if __name__ == '__main__':
    # 如果你报numpy.ufunc size changed错误，尝试更新一下Numpy的版本
    # pip install numpy==1.16.0
    full_connection()

'''
第1次的损失为2.292725，准确率为0.160000
第2次的损失为2.331739，准确率为0.020000
…………………………
第100次的损失为1.510217，准确率为0.730000
'''
