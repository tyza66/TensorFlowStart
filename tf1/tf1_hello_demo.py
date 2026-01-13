import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 去警告

def tensorflow_demo():
   """
   tensoflow的基本结构
   """
   # python版本
   a=2
   b=3
   c=a+b
   print('python版本的加法操作：\n',c) # 常规的python加法操作 能得到结果

   # tensorflow实现加法操作
   a_t=tf.constant(2) # 定义常量张量
   b_t=tf.constant(3)
   c_t=a_t+b_t
   print('tensorflow版本的加法操作：\n',c_t) # 直接打印的是张量的结构信息 需要开启会话才能得到结果

   # 开启会话
   with tf.Session() as sess:
      c_t_value=sess.run(c_t)
      print('开启会话的结果\n',c_t_value) # 开启会话之后才开始计算张量的值 能得到结果


if __name__ == '__main__':
   tensorflow_demo()
'''
python版本的加法操作：
 5
tensorflow版本的加法操作：
 Tensor("add:0", shape=(), dtype=int32)
开启会话的结果
 5
'''
