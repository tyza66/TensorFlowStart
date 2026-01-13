import tensorflow as tf

# 打印当前的 TF 版本
def print_tf_version():
    # 打印 TensorFlow 版本信息
    print("TensorFlow version:", tf.__version__)
    # 检查是否启用了 GPU 支持
    print("Is GPU available:", tf.config.list_physical_devices('GPU') != [])
