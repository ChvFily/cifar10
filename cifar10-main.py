import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from tensorflow import keras

def preprocess(x,y):
    x = 2 * tf.cast(x,dtype = tf.float32) / 255. - 1
    y = tf.cast(y,dtype = tf.int32)
    return x,y

batchsz = 128
## 加载数据库
(x,y),(x_test,y_test) = datasets.cifar10.load_data()
# print(x.shape,y_test.shape)
y = tf.squeeze(y) ##挤压文件
y_test = tf.squeeze(y_test)
# print(x.shape,y_test.shape)

y = tf.one_hot(y,depth=10)
y_test = tf.one_hot(y_test,depth=10)

##构建数据集
db_train  = tf.data.Dataset.from_tensor_slices((x,y))
# print("db_train",db_train)
db_train = db_train.map(preprocess).shuffle(10000).batch(batchsz)
# print("db_train-1",db_train)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
# print("db_train-2",db_test)
db_test = db_test.map(preprocess).batch(batchsz)
# print("db_train-3",db_test)

## 自定义层
class MyDense(layers.Layer):
    def __init__(self,inp_dim,out_dim):
        super(MyDense,self).__init__()
        self.kernal = self.add_weight('w',[inp_dim,out_dim]) ## 参数的设置
        self.bias = self.add_weight('b',[out_dim])

    def call(self,inputs,training = None):
        x = inputs @ self.kernal + self.bias
        return x

## 自定义网络结构
class MyNetWork(keras.Model):
    def __init__(self):
        super(MyNetWork,self).__init__()
        self.fc1 = MyDense(32*32*3,1024)
        self.fc2 = MyDense(1024,512)
        self.fc3 = MyDense(512,256)
        self.fc4 = MyDense(256,128)
        self.fc5 = MyDense(128,64)
        self.fc6 = MyDense(64,32)
        self.fc7 = MyDense(32,10)

    def call(self,inputs,training = None):
        """
        inputs :[b,32,32,3]
        train :[b,]
        """
        x = tf.reshape(inputs,[-1,32*32*3])

        x = self.fc1(x)
        x = tf.nn.relu(x)

        x = self.fc2(x)
        x = tf.nn.relu(x)

        x = self.fc3(x)
        x = tf.nn.relu(x)

        x = self.fc4(x)
        x = tf.nn.relu(x)

        x = self.fc5(x)
        x = tf.nn.relu(x)

        x = self.fc6(x)
        x = tf.nn.relu(x)

        x = self.fc7(x)

        return x

## 开始 初始化网络，开始训练  
network = MyNetWork()
network.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True)
                ,metrics=['accuracy'])
network.fit(db_train,epochs=10,validation_data=db_test,validation_freq=1)
network.evaluate(db_test) ## 测试模型
network.save_weights('ckpt/weights.ckpt')  ## 保存模型
del network  ## 销毁网路模型
print('saved to ckpt/weights.ckpt')


## 提取模型
network = MyNetWork()
network.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True)
                ,metrics=['accuracy'])
network.load_weights('ckpt/weights.ckpt')
print("loaded weight from  ckpt/weights.ckpt")

## 测试
print("test:")
network.evaluate(db_test) ## 数据与标签



