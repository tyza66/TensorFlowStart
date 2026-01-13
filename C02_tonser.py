import tensorflow as tf
import numpy as np
# 张量的值是不能改变的（不可变的），类似于 Python 中的元组（tuple）
# 创建0维张量（标量）
def create_scalar_tensor():
    scalar = tf.constant(5)
    # 打印标量张量
    print("Scalar tensor:", scalar)
    # 打印张量的维度
    print("Scalar tensor shape:", scalar.shape)
    # 打印变量的值
    print("Scalar tensor value:", scalar.numpy())

#  创建1维张量（向量）
def create_tensor_from_list():
    vector = tf.constant([1, 2, 3, 4, 5])
    # 打印向量张量
    print("Vector tensor:", vector)
    # 打印张量的维度
    print("Vector tensor shape:", vector.shape)
    # 打印变量的值
    print("Vector tensor values:", vector.numpy())

# 创建2维张量（矩阵）
def create_matrix_tensor():
    matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
    # 打印矩阵张量
    print("Matrix tensor:\n", matrix)
    # 打印张量的维度
    print("Matrix tensor shape:", matrix.shape)
    # 打印变量的值
    print("Matrix tensor values:\n", matrix.numpy())

# 创建3维张量
def create_3d_tensor():
    tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # 打印3维张量
    print("3D tensor:\n", tensor_3d)
    # 打印张量的维度
    print("3D tensor shape:", tensor_3d.shape)
    # 打印变量的值
    print("3D tensor values:\n", tensor_3d.numpy())

# 从 numpy 数组创建张量
def create_tensor_from_numpy():
    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_np = tf.constant(np_array)
    # 打印从 numpy 数组创建的张量
    print("Tensor from numpy array:\n", tensor_from_np)
    # 打印张量的维度
    print("Tensor from numpy array shape:", tensor_from_np.shape)
    # 打印变量的值
    print("Tensor from numpy array values:\n", tensor_from_np.numpy())

# 创建特殊张量
def create_special_tensors():
    # 创建全零张量、全一张量和随机张量
    zeros_tensor = tf.zeros((2, 3))
    # 创建全一张量
    ones_tensor = tf.ones((3, 2))
    # 创建随机张量
    random_tensor = tf.random.uniform((2, 2), minval=0, maxval=10) # 后面参数是最小值最大值
    # 创建单位矩阵
    eye = tf.eye(3)
    # 创建填充张量
    filled_tensor = tf.fill((2, 2), 7)


    # 打印特殊张量
    print("Zeros tensor:\n", zeros_tensor)
    print("Ones tensor:\n", ones_tensor)
    print("Random tensor:\n", random_tensor)
    print("Identity matrix:\n", eye)
    print("Filled tensor:\n", filled_tensor)

# 张量的属相
def tensor_properties():
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    print("Tensor:", tensor) # 张量本身
    print("Shape:", tensor.shape) # 张量的形状
    print("Data type:", tensor.dtype) # 张量的数据类型
    print("Num of dimensions:", tensor.ndim) # 维度信息（秩)
    print("Size:", tf.size(tensor).numpy()) # 张量中元素的数量
    print("Values:\n", tensor.device) # 张量的设备信息

# 张量的述职类型转换
def tensor_type_conversion():
    tensor = tf.constant([1.5, 2.5, 3.5], dtype=tf.float32)
    print("Original tensor:", tensor)
    # 转换为整数类型
    int_tensor = tf.cast(tensor, dtype=tf.int32)
    print("Converted to int tensor:", int_tensor)
    # 转换为双精度浮点数类型
    double_tensor = tf.cast(tensor, dtype=tf.float64)
    print("Converted to double tensor:", double_tensor)

# 张量的数学运算
def tensor_math_operations():
    tensor_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    tensor_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

    # 张量加法
    add_result = tf.add(tensor_a, tensor_b)
    print("Addition result:\n", add_result)

    # 张量减法
    sub_result = tf.subtract(tensor_b, tensor_a)
    print("Subtraction result:\n", sub_result)

    # 张量乘法
    mul_result = tf.multiply(tensor_a, tensor_b)
    print("Multiplication result:\n", mul_result)

    # 张量矩阵乘法
    matmul_result = tf.matmul(tensor_a, tensor_b)
    print("Matrix multiplication result:\n", matmul_result)

    # 张量除法
    div_result = tf.divide(tensor_b, tensor_a)
    print("Division result:\n", div_result)

    # 也可直接用 + - * / 运算符
    print("Addition using + operator:\n", tensor_a + tensor_b)
    print("Subtraction using - operator:\n", tensor_b - tensor_a)
    print("Multiplication using * operator:\n", tensor_a * tensor_b)
    print("Division using / operator:\n", tensor_b / tensor_a)

    # 还支持平方根 指数 对数等数学运算
    sqrt_result = tf.sqrt(tensor_a)
    print("Square root result:\n", sqrt_result)
    exp_result = tf.exp(tensor_a)
    print("Exponential result:\n", exp_result)
    log_result = tf.math.log(tensor_a)
    print("Logarithm result:\n", log_result)

# 张量的形状操作
# 不会改变从左上角到右下角的元素顺序
def tensor_shape_operations():
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    print("Original tensor shape:", tensor.shape)

    # 重塑张量形状
    reshaped_tensor = tf.reshape(tensor, (3, 2))
    print("Reshaped tensor:\n", reshaped_tensor)
    print("Reshaped tensor shape:", reshaped_tensor.shape)

    # 扩展张量维度
    expanded_tensor = tf.expand_dims(tensor, axis=0)
    print("Expanded tensor:\n", expanded_tensor)
    print("Expanded tensor shape:", expanded_tensor.shape)

    # 压缩张量维度
    squeezed_tensor = tf.squeeze(expanded_tensor)
    print("Squeezed tensor:\n", squeezed_tensor)
    print("Squeezed tensor shape:", squeezed_tensor.shape)

    # 展平
    flattened_tensor = tf.reshape(tensor, [-1]) # -1 表示自动计算该维度的大小
    print("Flattened tensor:\n", flattened_tensor)
    print("Flattened tensor shape:", flattened_tensor.shape)

    # 转置
    transposed_tensor = tf.transpose(tensor)
    print("Transposed tensor:\n", transposed_tensor)
    print("Transposed tensor shape:", transposed_tensor.shape)

    # 增加维度
    new_axis_tensor = tf.expand_dims(tensor, axis=0) # 在第0维增加一个新轴
    print("Tensor with new axis:\n", new_axis_tensor)
    print("Tensor with new axis shape:", new_axis_tensor.shape)

    # 减少维度
    reduced_axis_tensor = tf.squeeze(new_axis_tensor) # 去掉所有维度为1
    print("Tensor with reduced axis:\n", reduced_axis_tensor)
    print("Tensor with reduced axis shape:", reduced_axis_tensor.shape)

# 索引和切片
# 下标从0开始
def tensor_indexing_slicing():
    tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Original tensor:\n", tensor)

    # 索引单个元素
    element = tensor[1, 2] # 第二行第三列元素
    print("Indexed element:", element.numpy())

    # 切片获取子张量
    sub_tensor = tensor[0:2, 1:3] # 前两行，第二列到第三列
    print("Sliced sub-tensor:\n", sub_tensor.numpy())

    # 使用布尔掩码进行索引
    mask = tensor > 5
    masked_tensor = tf.boolean_mask(tensor, mask)
    print("Masked tensor (elements > 5):\n", masked_tensor.numpy())

# 张量广播
# 张量广播是指在进行张量运算时，自动扩展较小张量的形状以匹配较大张量的形状，从而使得它们能够进行逐元素的运算
# 通过逻辑扩展小张量的维度使其与大张量维度兼容
# 如果是行不够相当于在本身复制当前行到够数，如果是列不够相当于在本身复制当前列到够数
def tensor_broadcasting():
    tensor_a = tf.constant([[1, 2, 3], [4, 5, 6]]) # 2x3 张量
    tensor_b = tf.constant([10, 20, 30]) # 1x3 张量

    # 张量广播进行加法运算
    broadcasted_add = tf.add(tensor_a, tensor_b) # tensor_a + tensor_b
    print("Broadcasted addition result:\n", broadcasted_add)

    tensor_c = tf.constant([[1], [2], [3]])
    tensor_d = tf.constant([[10, 20, 30]])

    # 张量广播进行乘法运算
    broadcasted_mul = tf.multiply(tensor_c, tensor_d)
    print("Broadcasted multiplication result:\n", broadcasted_mul)
