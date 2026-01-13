import tensorflow as tf
import pandas as pd
import numpy as np
import glob

# 1、读取图片数据
# key, value = read(file_queue)
# key: 文件名 - labels.csv - 目标值value:一个样本的内容
def read_pic():
	#1、构建文件名队列
	file_names=glob.glob('./data/GenPics/*.jpg')
	file_queue=tf.train.string_input_producer(file_names)
	#2、读取与解码
	reader=tf.WholeFileReader()
	filename,image=reader.read(file_queue)
	#解码阶段
	decoded_image=tf.image.decode_jpeg(image)
	#更新形状，将图片形状确定下来
	decoded_image.set_shape([20,80,3])
	#修改图片的类型
	cast_image=tf.cast(decoded_image,tf.float32)
	#3、批处理
	filename_batch,image_batch=tf.train.batch([filename,cast_image],batch_size = 100,num_threads = 1,capacity = 200)
	return filename_batch,image_batch

# 2、解析CSV文件，建立文件名和标签值对应表格
def parse_csv():
	# 读取文件
	csv_data=pd.read_csv('./data/GenPics/labels.csv',names = ['file_num','chars'],index_col = 'file_num')
	#根据字母生成对应数字
	# NZPP  [13,25,15,15]
	labels=[]
	for label in csv_data['chars']:
		letter=[]
		for word in label:
			# A在26字母中的序号为0 B为1
			letter.append(ord(word)-ord('A'))  # ord() 返回对应的ASCII数值
		labels.append(letter)

	csv_data['labels']=labels  # 更新labels列的内容

	return csv_data

# 3、将一个样本的特征值和目标值一一对应
# 通过文件名查表(csv_data)
def filename2label(filenames,csv_data):
	labels=[]

	for file_name in filenames:
		file_num="".join((list(filter(str.isdigit,str(file_name))))) # 获取csv的序号，从而找到目标值
		target_labels=csv_data.loc[int(file_num),'labels']  # 目标值
		labels.append(target_labels)
	return np.array(labels)

# 4、建立卷积神经网络模型 == 》得出y_predict
def create_weights(shape):
	# 定义权重和偏置  stddev 标准差
	return tf.Variable(initial_value = tf.random_normal(shape = shape,stddev = 0.01))

def create_model(x):
	# 构建卷积神经网络模型
	# x: [None,20,80,3]
	# 1、第一卷积大层
	with tf.variable_scope('conv1'):
		# 卷积层
		# 定义32个filter和偏置
		conv1_weights=create_weights(shape = [5,5,3,32])
		conv1_bias=create_weights(shape = [32])
		conv1_x=tf.nn.conv2d(input = x,filter = conv1_weights,strides = [1,1,1,1],padding="SAME")+conv1_bias
		# 激活层
		relu1_x=tf.nn.relu(conv1_x)
		# 池化层
		pool1_x=tf.nn.max_pool(value =relu1_x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")

	# 2、第一卷积大层
	with tf.variable_scope('conv2'):
		# 卷积层
		# 定义64个filter和偏置
		conv2_weights = create_weights(shape = [5, 5, 32, 64])
		conv2_bias = create_weights(shape = [64])
		conv2_x = tf.nn.conv2d(input = pool1_x, filter = conv2_weights, strides = [1, 1, 1, 1],padding = "SAME") + conv2_bias
		# 激活层
		relu2_x = tf.nn.relu(conv2_x)
		# 池化层
		pool2_x = tf.nn.max_pool(value = relu2_x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	# 全连接层
	with tf.variable_scope('full_connection'):
		# 改变形状，修改为二维数组类型
		# # 输入[None,10,40,32]-->[None,5,20,64]
		#[None, 5, 20, 64]-->[None, 5 * 20 * 64]
		#[None,5 * 20 * 64]*[5 * 20 * 64,4 * 26] = [None,4 * 26]
		x_fc=tf.reshape(pool2_x,shape = [-1,5*20*64])
		weights_fc=create_weights(shape = [5*20*64,4*26])
		bias_fc=create_weights(shape = [4*26])
		y_predict=tf.matmul(x_fc,weights_fc)+bias_fc
	return y_predict

if __name__ == '__main__':
	filename,image=read_pic()
	csv_data=parse_csv()

	# 1、准备数据
	x=tf.placeholder(tf.float32,shape = [None,20,80,3])
	y_true=tf.placeholder(tf.float32,shape = [None,4*26])
	# 2、构建模型
	y_predict=create_model(x)
	# 3、构建损失函数
	loss_list=tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true,logits = y_predict)
	loss=tf.reduce_mean(loss_list)
	# 4、优化损失
	# optimizer=tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss) # 梯度下降，最小化
	optimizer=tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss) # Adam优化器，最小化

	# 5、计算准确率
	equal_list=tf.reduce_all(tf.equal(tf.argmax(tf.reshape(y_predict,shape = [-1,4,26]),axis = 2),
									  tf.argmax(tf.reshape(y_true, shape = [-1, 4,26]), axis = 2)),axis=1)
	accuracy=tf.reduce_mean(tf.cast(equal_list,tf.float32))
	# 初始化变量
	init=tf.global_variables_initializer()

	# 开启会话
	with tf.Session() as sess:
		sess.run(init)
		# 开启线程
		coor=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(sess = sess,coord = coor)
		for i in range(150):
			filename_value,image_value=sess.run([filename,image])
			labels=filename2label(filename_value,csv_data) # 将一个样本的特征值和目标值一一对应
			# 将目标值转换为one-hot编码
			labels_value=tf.reshape(tf.one_hot(labels,depth = 26),[-1,4*26]).eval()
			_,error,accuracy_value=sess.run([optimizer,loss,accuracy],feed_dict = {x:image_value,y_true:labels_value})
			print('第%d次训练损失为%f,准确率为%f'%(i+1,error,accuracy_value))

		# 回收线程
		coor.request_stop()
		coor.join(threads)
