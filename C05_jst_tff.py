import tensorflow as tf

# 在TF2中，计算图和 tf.function的概念被大大的简化和自动化，TF2中引入了立即执行模式（Eager Execution），使得代码更直观易懂。
# 然而，对于性能优化和复杂模型，计算图仍然是重要的概念。tf.function允许用户将Python函数转换为计算图，从而提升性能。

# tf.function是一个装饰器，用于将Python函数转换为TensorFlow计算图 可以在训练和推理中使用
@tf.function
def simple_add_function(x, y):
    return x + y

# 执行加法操作并计算图
def execute_tf_function():
    a = tf.constant(3)
    b = tf.constant(4)
    result = simple_add_function(a, b)
    print("Result of tf.function addition:", result) # 能计算出结果 但不是立即执行的 而是计算图之后再计算的

