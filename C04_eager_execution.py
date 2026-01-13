import tensorflow as tf
# 立即执行模式
# 相比于TF1中的图模式无需构建图的会话了
# 开发这可以实时查看变量的值和操作结果，TF2默认开启
def eager_execution_demo():
    # 检查是否启用了立即执行模式
    print("Eager execution is enabled:", tf.executing_eagerly())

    # 创建两个常量张量
    a = tf.constant(3)
    b = tf.constant(4)

    # 执行加法操作
    c = a + b

    # 打印结果 (TF2直接可以计算出结果，而在TF1中需要在会话中运行才行)
    print("Result of addition:", c.numpy())
