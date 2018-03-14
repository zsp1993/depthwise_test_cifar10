# -*- coding: utf-8 -*-

from model.model_ops import *
import model.input_dataset
import time
import os
import tarfile
from tensorflow.python.framework import ops

#dropoup参数
keep_prob = tf.placeholder("float")
#学习率
learn_rate = tf.placeholder("float")
#样本大小
picture = tf.placeholder("float", shape=[None,24,24,3])

class MyModel():
    def __init__(self,looP=300000,input_Channel=3,layer1_Node_num=32,layer2_Node_num=64,
                layer3_Node_num=128,fulllayer1_Node_num=128,fulllayer2_Node_num=128,batch_Size=128,
                depthwise_model_Save_path='mynet2/save_net.ckpt',gen_model_Save_path='mynet1/save_net.ckpt'):

        self.input_channel = input_Channel
        self.layer1_node_num = layer1_Node_num
        self.layer2_node_num = layer2_Node_num
        self.layer3_node_num = layer3_Node_num
        self.fulllayer1_node_num = fulllayer1_Node_num
        self.fulllayer2_node_num = fulllayer2_Node_num
        self.depthwise_model_save_path = depthwise_model_Save_path
        self.gen_model_save_path = gen_model_Save_path
        self.batch_size = batch_Size
        self.loop = looP

    def genModel(self,x,y):
        #第一卷积层（3 #24x24->16 #24x24）
        w_conv1 = weight_variable([3, 3, self.input_channel, self.layer1_node_num],"w_conv1")
        b_conv1 = bias_variable([self.layer1_node_num],"b_conv1")
        # 而后，我们利用ReLU激活函数，对其进行第一次卷积。
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1, [1, 1, 1, 1]) + b_conv1,name="h_conv1")


        # 第一次池化(16 #24x24->16 #24x24)
        # 比较容易理解，使用2x2的网格以max pooling的方法池化。
        #h_pool1 = max_pool_2x2(h_conv1)
        h_pool1 = h_conv1

        # 第二层卷积(16 #24x24->32 #24x24)
        w_conv2 = weight_variable([2, 2, self.layer1_node_num, self.layer2_node_num],"w_conv2")
        b_conv2 = bias_variable([self.layer2_node_num],"b_conv2")

        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, [1, 1, 1, 1]) + b_conv2,name="h_conv2")
        #h_pool2 = max_pool_2x2(h_conv2)
        h_pool2 = h_conv2

        # 第三层卷积与第三次池化(32 #24x24->64 #24x24->64 #24x24)
        # 与第一层卷积、第一次池化类似的过程。
        w_conv3 = weight_variable([2, 2, self.layer2_node_num, self.layer3_node_num],"w_conv3")
        b_conv3 = bias_variable([self.layer3_node_num],"b_conv3")

        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, [1, 1, 1, 1]) + b_conv3,name="h_conv3")
        #h_pool3 = max_pool_2x2(h_conv3)
        h_pool3 = h_conv3

        # 密集连接层
        # 此时，图片是8x8的大小。我们在这里加入一个有256个神经元的全连接层。
        # 之后把刚才池化后输出的张量reshape成一个一维向量，再将其与权重相乘，加上偏置项，再通过一个ReLU激活函数。
        w_fc1 = weight_variable([24 * 24 * self.layer3_node_num, self.fulllayer1_node_num],"w_fc1")
        b_fc1 = bias_variable([self.fulllayer1_node_num],"b_fc1")

        h_pool3_flat = tf.reshape(h_pool3, [-1, 24 * 24 * self.layer3_node_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

        #第二密集连接层
        w_fc2 = weight_variable([self.fulllayer1_node_num, self.fulllayer2_node_num],"w_fc2")
        b_fc2 = bias_variable([self.fulllayer2_node_num],"b_fc2")

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

        # Dropout 这是一个比较新的也非常好用的防止过拟合的方法
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        # 回归与输出 应用了简单的softmax，输出。
        w_fc3 = weight_variable([self.fulllayer2_node_num, 10],"w_fc3")
        b_fc3 = bias_variable([10],"b_fc3")

        # y_conv = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)

        #定义损失函数
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=y,name='likelihood_loss')
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')

        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y,name='likelihood_loss'))
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy_loss)

        #模型评估
        #转化为onehot
        size1 = tf.size(y)
        y = tf.expand_dims(y, 1)
        indices = tf.expand_dims(tf.range(0, size1, 1), 1)
        concated = tf.concat([indices, y], 1)
        y = tf.sparse_to_dense(concated, tf.stack([size1, 10]), 1.0, 0.0)
        correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return cross_entropy_loss,optimizer,accuracy

    def depthwiseModel(self,x,y,save_path):

        # 第一卷积层（3 #24x24->16 #24x24）
        w_conv1 = weight_variable([3, 3, self.input_channel, self.layer1_node_num], "w_conv1")
        b_conv1 = bias_variable([self.layer1_node_num], "b_conv1")
        # 而后，我们利用ReLU激活函数，对其进行第一次卷积。
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1, [1, 1, 1, 1]) + b_conv1, name="h_conv1")
        '''
        # 第一卷积层（3 #24x24->12 #24x24->32 #24x24）
        num_bias = 1
        w_conv11 = weight_variable([3, 3, self.input_channel, num_bias],"w_conv11")
        b_conv11 = bias_variable([self.input_channel*num_bias],"b_conv11")
        # 而后，我们利用ReLU激活函数，对其进行第一次depthwise卷积。
        h_conv1 = tf.nn.relu(depthwise_conv2d(x, w_conv11, [1, 1, 1, 1]) + b_conv11)
        w_conv12 = weight_variable([1,1,self.input_channel*num_bias,self.layer1_node_num],"w_conv12")
        b_conv12 = bias_variable([self.layer1_node_num],"b_conv12")
        h_conv1 = tf.nn.relu(conv2d(h_conv1, w_conv12, [1, 1, 1, 1]) + b_conv12)
        '''

        #第二卷积层（32#24x24 -> 32#24*24 ->64#24*24）
        num_bias = 1
        w_conv21 = weight_variable([2, 2, self.layer1_node_num, num_bias],"w_conv21")
        b_conv21 = bias_variable([self.layer1_node_num * num_bias],"b_conv21")
        # 而后，我们利用ReLU激活函数，对其进行第一次depthwise卷积。
        h_conv2 = tf.nn.relu(depthwise_conv2d(h_conv1, w_conv21, [1, 1, 1, 1]) + b_conv21)
        w_conv22 = weight_variable([1, 1, self.layer1_node_num * num_bias, self.layer2_node_num],"w_conv22")
        b_conv22 = bias_variable([self.layer2_node_num],"b_conv22")
        h_conv2 = tf.nn.relu(conv2d(h_conv2, w_conv22, [1, 1, 1, 1]) + b_conv22)

        '''
        # 第二层卷积(16 #24x24->32 #24x24)
        w_conv2 = weight_variable([2, 2, self.layer1_node_num, self.layer2_node_num],"w_conv2")
        b_conv2 = bias_variable([self.layer2_node_num],"b_conv2")

        h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, [1, 1, 1, 1]) + b_conv2)
        '''
        h_pool2 = h_conv2


        # 第三卷积层（64#24x24 -> 64#24*24 ->128#24*24）
        num_bias = 1
        w_conv31 = weight_variable([2, 2, self.layer2_node_num, num_bias],"w_conv31")
        b_conv31 = bias_variable([self.layer2_node_num * num_bias],"b_conv31")
        # 而后，我们利用ReLU激活函数，对其进行第一次depthwise卷积。
        h_conv3 = tf.nn.relu(depthwise_conv2d(h_conv2, w_conv31, [1, 1, 1, 1]) + b_conv31)
        w_conv32 = weight_variable([1, 1, self.layer2_node_num * num_bias, self.layer3_node_num],"w_conv32")
        b_conv32 = bias_variable([self.layer3_node_num],"b_conv32")
        h_conv3 = tf.nn.relu(conv2d(h_conv3, w_conv32, [1, 1, 1, 1]) + b_conv32)

        '''
        # 第三层卷积与第三次池化(32 #24x24->64 #24x24->64 #24x24)
        # 与第一层卷积、第一次池化类似的过程。
        w_conv3 = weight_variable([2, 2, self.layer2_node_num, self.layer3_node_num],"w_conv3")
        b_conv3 = bias_variable([self.layer3_node_num],"b_conv3")

        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, [1, 1, 1, 1]) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)
        '''

        h_pool3 = h_conv3

        # 密集连接层
        # 此时，图片是8x8的大小。我们在这里加入一个有256个神经元的全连接层。
        # 之后把刚才池化后输出的张量reshape成一个一维向量，再将其与权重相乘，加上偏置项，再通过一个ReLU激活函数。
        w_fc1 = weight_variable([24 * 24 * self.layer3_node_num, self.fulllayer1_node_num],"w_fc1")
        b_fc1 = bias_variable([self.fulllayer1_node_num],"b_fc1")

        h_pool3_flat = tf.reshape(h_pool3, [-1, 24 * 24 * self.layer3_node_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

        # 第二密集连接层
        w_fc2 = weight_variable([self.fulllayer1_node_num, self.fulllayer2_node_num],"w_fc2")
        b_fc2 = bias_variable([self.fulllayer2_node_num],"b_fc2")

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

        # Dropout 这是一个比较新的也非常好用的防止过拟合的方法
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        # 回归与输出 应用了简单的softmax，输出。
        w_fc3 = weight_variable([self.fulllayer2_node_num, 10],"w_fc3")
        b_fc3 = bias_variable([10],"b_fc3")

        # y_conv = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)

        # 定义损失函数
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=y,
                                                                            name='likelihood_loss')
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')

        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y,name='likelihood_loss'))
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross_entropy_loss)
        '''
        #是否只训练变量子集
        opt = tf.train.MomentumOptimizer(learn_rate,0.9)
        grads = opt.compute_gradients(cross_entropy_loss,var_list=[w_conv31]+[b_conv31]+[w_conv32]+[b_conv32])
        optimizer = opt.apply_gradients(grads)
        '''
        # 模型评估
        # 转化为onehot
        size1 = tf.size(y)
        y = tf.expand_dims(y, 1)
        indices = tf.expand_dims(tf.range(0, size1, 1), 1)
        concated = tf.concat([indices, y], 1)
        y = tf.sparse_to_dense(concated, tf.stack([size1, 10]), 1.0, 0.0)
        correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return cross_entropy_loss, optimizer, accuracy,w_conv1,b_conv1,w_fc1,b_fc1,w_fc2,b_fc2,w_fc3,b_fc3


    def train(self,data_path,depwise=True):

        # 读取数据
        img_batch, label_batch = model.input_dataset.preprocess_input_data(data_path,self.batch_size)  # 输入图像的预处理，包括亮度、对比度、图像翻转等操作
        label_batch = tf.cast(x=label_batch, dtype=tf.int32)  # 强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求
        if depwise == False:
            cost,optimizer,accuracy= self.genModel(img_batch,label_batch)
            model_save_path = self.gen_model_save_path
        else:
            cost, optimizer, accuracy ,w_conv1,b_conv1,w_fc1,b_fc1,w_fc2,b_fc2,w_fc3,b_fc3= self.depthwiseModel(img_batch, label_batch,self.gen_model_save_path)
            model_save_path = self.depthwise_model_save_path
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if depwise==True:
                saver1 = tf.train.Saver([w_conv1]+[b_conv1]+[w_fc1]+[b_fc1]+[w_fc2]+[b_fc2]+[w_fc3]+[b_fc3])
                save_path = saver1.restore(sess, self.gen_model_save_path)
            # 线程需要同步
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord)
            # threads = tf.train.start_queue_runners(sess=sess)
            new_learn_rate = 1e-4
            rate = 10 ** (-4.0 / (self.loop / 2000))
            #print(sess.run(w_conv2))
            #t0 = time.time()
            for i in range(self.loop):
                if (i + 1) % 2000 == 0:
                    new_learn_rate = new_learn_rate * rate
                    save_path = saver.save(sess, model_save_path)
                    # 输出保存路径
                    print('Save to path: ', save_path)
                # with tf.device("/gpu:0"):
                # mse,_=sess.run([model,train_step],feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})
                #if i%30==1:
                #    t0 = time.time()
                if i % 30 == 0:

                    cost1,_,accuracy1 = sess.run([cost, optimizer,accuracy], feed_dict={keep_prob: 1.0, learn_rate: new_learn_rate})
                    # print(sess.run(y_conv, feed_dict={keep_prob: 1}))
                    print("step %d,cost is %g, training accuracy %g" % (i,cost1, accuracy1))
                    print("###########")
                    #print(sess.run(w_conv2))

                else:

                    _,_,accuracy1 = sess.run([cost, optimizer,accuracy], feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})

            coord.request_stop()
            coord.join(threads)

    def depthwise_test(self,data_path):
        # 读取数据
        img, y = model.input_dataset.preprocess_input_data(data_path,100,True)  # 输入图像的预处理，包括亮度、对比度、图像翻转等操作
        #label = tf.cast(x=label,dtype=tf.int32)  # 强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求
        y = tf.cast(x=y, dtype=tf.int32)
        # 第一卷积层（3 #24x24->16 #24x24）
        w_conv1 = weight_variable([3, 3, self.input_channel, self.layer1_node_num], "w_conv1")
        b_conv1 = bias_variable([self.layer1_node_num], "b_conv1")
        # 而后，我们利用ReLU激活函数，对其进行第一次卷积。
        h_conv1 = tf.nn.relu(conv2d(img, w_conv1, [1, 1, 1, 1]) + b_conv1, name="h_conv1")
        '''
        # 第一卷积层（3 #24x24->12 #24x24->32 #24x24）
        num_bias = 1
        w_conv11 = weight_variable([3, 3, self.input_channel, num_bias],"w_conv11")
        b_conv11 = bias_variable([self.input_channel * num_bias],"b_conv11")
        # 而后，我们利用ReLU激活函数，对其进行第一次depthwise卷积。
        h_conv1 = tf.nn.relu(depthwise_conv2d(img, w_conv11, [1, 1, 1, 1]) + b_conv11)
        w_conv12 = weight_variable([1, 1, self.input_channel * num_bias, self.layer1_node_num],"w_conv12")
        b_conv12 = bias_variable([self.layer1_node_num],"b_conv12")
        h_conv1 = tf.nn.relu(conv2d(h_conv1, w_conv12, [1, 1, 1, 1]) + b_conv12)
        '''

        # 第二卷积层（32#24x24 -> 32#24*24 ->64#24*24）
        num_bias = 1
        w_conv21 = weight_variable([2, 2, self.layer1_node_num, num_bias], "w_conv21")
        b_conv21 = bias_variable([self.layer1_node_num * num_bias], "b_conv21")
        # 而后，我们利用ReLU激活函数，对其进行第一次depthwise卷积。
        h_conv2 = tf.nn.relu(depthwise_conv2d(h_conv1, w_conv21, [1, 1, 1, 1]) + b_conv21)
        w_conv22 = weight_variable([1, 1, self.layer1_node_num * num_bias, self.layer2_node_num], "w_conv22")
        b_conv22 = bias_variable([self.layer2_node_num], "b_conv22")
        h_conv2 = tf.nn.relu(conv2d(h_conv2, w_conv22, [1, 1, 1, 1]) + b_conv22)

        '''
        # 第二卷积层（32#24x24 -> 32#24*24 ->64#24*24）
        w_conv2 = weight_variable([2, 2, self.layer1_node_num, self.layer2_node_num],"w_conv2")
        b_conv2 = bias_variable([self.layer2_node_num],"b_conv2")
        h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, [1, 1, 1, 1]) + b_conv2)
        '''
        h_pool2 = h_conv2

        '''
        # 第三层卷积与第三次池化(32 #24x24->64 #24x24->64 #24x24)
        # 与第一层卷积、第一次池化类似的过程。
        w_conv3 = weight_variable([2, 2, self.layer2_node_num, self.layer3_node_num],"w_conv3")
        b_conv3 = bias_variable([self.layer3_node_num],"b_conv3")

        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, [1, 1, 1, 1]) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)
        '''
        # 第三卷积层（64#24x24 -> 64#24*24 ->128#24*24）
        num_bias = 1
        w_conv31 = weight_variable([2, 2, self.layer2_node_num, num_bias], "w_conv31")
        b_conv31 = bias_variable([self.layer2_node_num * num_bias], "b_conv31")
        # 而后，我们利用ReLU激活函数，对其进行第一次depthwise卷积。
        h_conv3 = tf.nn.relu(depthwise_conv2d(h_conv2, w_conv31, [1, 1, 1, 1]) + b_conv31)
        w_conv32 = weight_variable([1, 1, self.layer2_node_num * num_bias, self.layer3_node_num], "w_conv32")
        b_conv32 = bias_variable([self.layer3_node_num], "b_conv32")
        h_conv3 = tf.nn.relu(conv2d(h_conv3, w_conv32, [1, 1, 1, 1]) + b_conv32)

        h_pool3 = h_conv3

        # 密集连接层
        # 此时，图片是8x8的大小。我们在这里加入一个有256个神经元的全连接层。
        # 之后把刚才池化后输出的张量reshape成一个一维向量，再将其与权重相乘，加上偏置项，再通过一个ReLU激活函数。
        w_fc1 = weight_variable([24 * 24 * self.layer3_node_num, self.fulllayer1_node_num],"w_fc1")
        b_fc1 = bias_variable([self.fulllayer1_node_num],"b_fc1")

        h_pool3_flat = tf.reshape(h_pool3, [-1, 24 * 24 * self.layer3_node_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

        # 第二密集连接层
        w_fc2 = weight_variable([self.fulllayer1_node_num, self.fulllayer2_node_num],"w_fc2")
        b_fc2 = bias_variable([self.fulllayer2_node_num],"b_fc2")

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        # 回归与输出 应用了简单的softmax，输出。
        w_fc3 = weight_variable([self.fulllayer2_node_num, 10],"w_fc3")
        b_fc3 = bias_variable([10],"b_fc3")

        # y_conv = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)

        #y_conv = tf.argmax(y_conv, 1)
        # 转化为onehot
        size1 = tf.size(y)
        y = tf.expand_dims(y, 1)
        indices = tf.expand_dims(tf.range(0, size1, 1), 1)
        concated = tf.concat([indices, y], 1)
        y = tf.sparse_to_dense(concated, tf.stack([size1, 10]), 1.0, 0.0)
        correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
            saver1 = tf.train.Saver()
            save_path = saver1.restore(sess, self.depthwise_model_save_path)

            # 线程需要同步
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord)
            print("depthwise testing#########################")
            #print(sess.run(w_conv2))

            for j in range(10):
                t0 = time.time()
                test_accuracy = 0
                for i in range(100):
                    accuracy1 = sess.run([accuracy],feed_dict={keep_prob:1})
                    test_accuracy = test_accuracy+accuracy1[0]
                test_accuracy = test_accuracy/10000
                print ("test accuracy is %g,time is %g" % (test_accuracy,time.time()-t0))

            coord.request_stop()
            coord.join(threads)

    def gen_test(self,data_path):
        # 读取数据
        img, y = model.input_dataset.preprocess_input_data(data_path,100,True)  # 输入图像的预处理，包括亮度、对比度、图像翻转等操作
        y = tf.cast(x=y,dtype=tf.int32)  # 强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求

        # 第一卷积层（3 #24x24->16 #24x24）
        w_conv1 = weight_variable([3, 3, self.input_channel, self.layer1_node_num],"w_conv1")
        b_conv1 = bias_variable([self.layer1_node_num],"b_conv1")
        # 而后，我们利用ReLU激活函数，对其进行第一次卷积。
        h_conv1 = tf.nn.relu(conv2d(img, w_conv1, [1, 1, 1, 1]) + b_conv1)

        # 第一次池化(16 #24x24->16 #24x24)
        # 比较容易理解，使用2x2的网格以max pooling的方法池化。
        # h_pool1 = max_pool_2x2(h_conv1)
        h_pool1 = h_conv1

        # 第二层卷积(16 #24x24->32 #24x24)
        w_conv2 = weight_variable([2, 2, self.layer1_node_num, self.layer2_node_num],"w_conv2")
        b_conv2 = bias_variable([self.layer2_node_num],"b_conv2")

        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, [1, 1, 1, 1]) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)
        h_pool2 = h_conv2

        # 第三层卷积与第三次池化(32 #24x24->64 #24x24->64 #24x24)
        # 与第一层卷积、第一次池化类似的过程。
        w_conv3 = weight_variable([2, 2, self.layer2_node_num, self.layer3_node_num],"w_conv3")
        b_conv3 = bias_variable([self.layer3_node_num],"b_conv3")

        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, [1, 1, 1, 1]) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)
        h_pool3 = h_conv3

        # 密集连接层
        # 此时，图片是8x8的大小。我们在这里加入一个有256个神经元的全连接层。
        # 之后把刚才池化后输出的张量reshape成一个一维向量，再将其与权重相乘，加上偏置项，再通过一个ReLU激活函数。
        w_fc1 = weight_variable([24 * 24 * self.layer3_node_num, self.fulllayer1_node_num],"w_fc1")
        b_fc1 = bias_variable([self.fulllayer1_node_num],"b_fc1")

        h_pool3_flat = tf.reshape(h_pool3, [-1, 24 * 24 * self.layer3_node_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

        # 第二密集连接层
        w_fc2 = weight_variable([self.fulllayer1_node_num, self.fulllayer2_node_num],"w_fc2")
        b_fc2 = bias_variable([self.fulllayer2_node_num],"b_fc2")

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

        # Dropout 这是一个比较新的也非常好用的防止过拟合的方法
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        # 回归与输出 应用了简单的softmax，输出。
        w_fc3 = weight_variable([self.fulllayer2_node_num, 10],"w_fc3")
        b_fc3 = bias_variable([10],"b_fc3")

        # y_conv = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)
        #y_conv = tf.argmax(y_conv, 1)
        size1 = tf.size(y)
        y = tf.expand_dims(y, 1)
        indices = tf.expand_dims(tf.range(0, size1, 1), 1)
        concated = tf.concat([indices, y], 1)
        y = tf.sparse_to_dense(concated, tf.stack([size1, 10]), 1.0, 0.0)
        correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
            saver2 = tf.train.Saver()
            save_path = saver2.restore(sess, self.gen_model_save_path)
            # 线程需要同步
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord)
            print("gen testing###############################")
            #print(sess.run(w_conv2))
            for j in range(10):
                t0 = time.time()
                test_accuracy = 0
                for i in range(100):
                    accuracy1 = sess.run([accuracy],feed_dict={keep_prob:1.0})
                    #print(accuracy1)
                    test_accuracy = test_accuracy+accuracy1[0]

                test_accuracy = test_accuracy/10000
                print ("test accuracy is %g,time is %g" % (test_accuracy,time.time()-t0))

            coord.request_stop()
            coord.join(threads)