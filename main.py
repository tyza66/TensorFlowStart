# By tyza66
# https://github.com/tyza66
#
from C01_HelloTF import print_tf_version
from C02_tonser import *
from C03_Variables import *
from C04_eager_execution import eager_execution_demo
from C05_jst_tff import execute_tf_function
from C06_automatic import automatic_differentiation_demo

# 第一节：Hello TensorFlow
def C01():
    print_tf_version()

# 第二节：张量
def C02():
    create_scalar_tensor()
    create_tensor_from_list()
    create_matrix_tensor()
    create_3d_tensor()
    create_tensor_from_numpy()
    create_special_tensors()
    tensor_properties()
    tensor_type_conversion()
    tensor_math_operations()
    tensor_shape_operations()

# 第三节：Variables
def C03():
    create_variable()
    create_2d_variable()

# 第四节：立即执行模式
def C04():
    eager_execution_demo()

# 第五节：计算图和tf.function
def C05():
    execute_tf_function()

# 第六节：自动微分
def C06():
    automatic_differentiation_demo()

if __name__ == '__main__':
    # 要玩那个就调用哪个
    C06() # 从C06后半段在jupyter里玩起
