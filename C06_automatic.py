import tensorflow as tf


# 自动微分和梯度
# 梯度是指函数对其输入的变量的倒数，它描述了函数在某一点的变化速度，在机器学习中，体服通常指的是损失函数对模型参数的导数
# 自动微分是一种通过计算图来自动求取梯度的技术，TF2通过tf.GradientTape实现自动微分,它可以追踪操作并计算梯度，使用这种方法可以高效地进行反向传播，进而计算损失函数对每个参数的梯度

# 定义一个简单的函数
def simple_fang_function(x):
    return x ** 2


# 使用自动微分计算梯度
def compute_gradient(x_value):
    # 使用tf.GradientTape来记录操作
    with tf.GradientTape() as tape:
        # 将输入变量转换为张量并启用梯度追踪
        x = tf.Variable(x_value, dtype=tf.float32)
        # 计算函数值
        y = simple_fang_function(x)
    # 计算y对x的梯度
    dy_dx = tape.gradient(y, x)
    print(f"Function value at x={x_value}: {y.numpy()}")
    print(f"Gradient at x={x_value}: {dy_dx.numpy()}")


# 测试自动微分
def automatic_differentiation_demo():
    x_test_value = 3.0
    compute_gradient(x_test_value)


from matplotlib import pyplot as plt

# 定义一个简单的线性模型
def linear_model(x):
    # 设置matplotlib使用黑体显示中文
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    # 正确的权重和偏置
    TRUE_W = 3.0
    TRUE_B = 2.0
    # 指定数据数量
    NUM_EXAMPLES = 201

    x = tf.linspace(-2, 2, NUM_EXAMPLES)  # 生成-2到2之间的201个均匀分布的
    x = tf.cast(x, tf.float32)  # 转换为float32类型

    # 生成噪声数据
    noise = tf.random.normal(shape=[NUM_EXAMPLES], mean=0.0, stddev=1.0)  # 添加噪声

    # 定义一个函数
    def f(x):
        return x * TRUE_W + TRUE_B

    y = f(x) + noise  # 完整定义y值

    # 转移至 ipynb

