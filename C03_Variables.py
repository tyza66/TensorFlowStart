import tensorflow as tf
import numpy as np

# Variable 是可以改变的张量，类似于 Python 中的列表（list）
def create_variable():
    # 创建一个初始值为 0 的变量
    var = tf.Variable(0)
    print("Initial variable value:", var.numpy())

    # 修改变量的值
    var.assign(10)
    print("Updated variable value:", var.numpy())

    # 增加变量的值
    var.assign_add(5)
    print("After adding 5, variable value:", var.numpy())

    # 减少变量的值
    var.assign_sub(3)
    print("After subtracting 3, variable value:", var.numpy())

# 定义二维的
def create_2d_variable():
    # 创建一个 2x3 的变量，初始值为随机数
    var_2d = tf.Variable(tf.random.uniform((2, 3)))
    print("Initial 2D variable:\n", var_2d.numpy())

    # 修改变量的值
    new_values = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    var_2d.assign(new_values)
    print("Updated 2D variable:\n", var_2d.numpy())

# 定义三维的
def create_3d_variable():
    # 创建一个 2x2x2 的变量，初始值为随机数
    var_3d = tf.Variable(tf.random.uniform((2, 2, 2)))
    print("Initial 3D variable:\n", var_3d.numpy())

    # 修改变量的值
    new_values = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
    var_3d.assign(new_values)
    print("Updated 3D variable:\n", var_3d.numpy())

