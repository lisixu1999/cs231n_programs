# Lecture2 Image Classification Pipeline

## 1. 图片在计算机中的储存方式&图像识别

### 1.1 图片在计算机中的储存方式
&nbsp; &nbsp; 计算机中的图片有两种，位图和矢量图，其中矢量图储存的是该图片的计算方式，比如在这张图距离顶部10%，距离右边20%的地方有一个点，哪里有一条线，线有多长等等。从这里我们们可以看出，矢量图放大是没有损失的。但是位图不一样。位图储存的是像素点，也就是说把图片当作矩阵来存储。

&nbsp; &nbsp; 我们知道，红黄蓝三种颜色混合可以构成不同的颜色。因此我们可以用三个矩阵来储存一张图片，即RGB三色，我们用一个0-255的数字表示这个颜色的深度，比如说一个像素点可以是R110，G250，B120混合而成

### 1.2 图像识别
&nbsp; &nbsp; 图像识别顾名思义，就是把图片中的内容提取出来。从Image 到 Label。

&nbsp; &nbsp; 其实很多任务都可以归结为，给一个输入，给一个输出（或者多个输出）。也就是说，我们需要找到一个函数，这个函数在给定输入图片的像素矩阵之后，可以输出这张照片是什么，比如是一只猫。

&nbsp; &nbsp; 对于我们人类来说，看一眼就知道了，但是电脑毕竟需要我们来编程他不会自己想（起码现在不会）如何让电脑来识别猫呢？

&nbsp; &nbsp; 我们可以尝试自己写出一个函数来识别猫吗？这……有点困难。当然我们也不会这么做。我们想让计算机来自己学习出猫长什么样，就像人一样。我们是怎么知道一只动物是猫的呢？我们见过很多相似的动物（train_image）同时又有别人告诉我们这是猫（train_label）于是，当我们再见到一个动物的时候，我们可以根据之前的“训练”来判断我们面前的是不是猫。

&nbsp; &nbsp; 因此我们希望也能对计算机执行类似的操作，我们想要训练一个模型，基本步骤如下：
* 输入：输入是包含N个图像的集合，每个图像的标签是K种分类标签中的一种。这个集合称为训练集（train_set）。
* 学习：这一步的任务是使用训练集来学习每个类到底长什么样。一般该步骤叫做训练分类器或者学习一个模型。
* 评价：让分类器来预测它未曾见过的图像（验证集 validation_set）的分类标签，并以此来评价分类器的质量。我们会把分类器预测的标签和图像真正的分类标签对比。毫无疑问，分类器预测的分类标签和图像真正的分类标签如果一致，那就是好事，这样的情况越多越好。

## 2. Nearest Neighbor 分类器
### 2.1 Nearest Neighbor 分类器
&nbsp; &nbsp; 我们知道，判断两张照片里面的是不是同一种生物，最简单的方法就是对比和两张图片的相似度，长得像，就应该是一个，Nearest Neighbor 分类器正是这么想的。我们首先要定义一个距离函数，来度量两张图片之间的距离，距离越近越有可能是同类。因此，我们对于一张图片，计算他和训练集所有图片之间的距离，取最近距离图片的标签来作为这张图片的标签。
### 2.2 KNN
&nbsp; &nbsp; 但是上面的那种方法有点不足，就是可能会出现个别特例，万一，有只狗像猫，我们送进去的猫的图片还就特别像怎么办？所以，我们改进上述方案，不只用一个了，用好多个，我们取K个离输入图片距离最近的训练图片，投票，那个label最多，就是哪个。
![分类可视化的结果](https://upload-images.jianshu.io/upload_images/9592233-34f1534ddc3de43a.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 2.3 超参数
&nbsp; &nbsp; 这是我们第一次面临选择，距离？图片的距离？图片有什么距离啊。我们已经知道图片在计算机中使用像素点存储，一种比较自然的想法就是，所有像素点数值相减取绝对值加和。这种距离称为L1距离（曼哈顿距离）。
* 曼哈顿距离 Manhattan Distance（L1）
 $$d_1(I_1,I_2) = \sum \left|I_1^p - I_2^p \right|  $$
* 欧氏距离 Euclidean Distance （L2）
 $$d_1(I_1,I_2) =\sqrt{ \sum  (I_1^p - I_2^p  )^2} $$
&nbsp; &nbsp;为什么叫做超参数呢，就是因为这两种哪种好其实谁都不知道，这个是试出来的有兴趣的可以试试，我给的代码里是L2距离。

&nbsp; &nbsp; 还有一个选择是K，K取多少比较合适呢？这个……自己慢慢试去吧。

### 2.4 数据集

数据集可以分为以下几个类：训练集(train_set)，验证集(validation_set)，测试集(test_set)。
* 训练集：用来训练模型，得出参数（过一会将出现的权重）的。
* 验证集：用来调超参数的。
* 测试集：评估最后结果的。
&nbsp; &nbsp;注意，训练集只是整个数据集的一部分，因为如果训练集包括了所有的数据，那么他们可能在这个数据集上表现良好，换一个就变差了，毕竟，这是用这个数据集训练出来的（过拟合）。
&nbsp; &nbsp;但是有时候数据较少，没有那么多可以划分，这时，可以使用交叉验证。我们将一份数据分成N份，前N-1份用于训练，第N份用于验证，循环着来，取N次验证的平均值做最后结果。

## 3. 线性回归
### 3.1 开始“学习”
&nbsp; &nbsp;从KNN中我们可以看出去，其实这个算法是没有在“学习”的，用我的话来说，KNN只是跟过往的经验来从表层判断，没有从中抽象出一般规律。这一点表现在，KNN的“训练”只是简单的把训练集输入，没有任何参数改变。我们可以想象，这种方法在处理未知图片时正确率一定不理想。我们希望从训练集中抽象出一般规律，这样可以更好地应对未知的情况。
### 3.2 线性回归
&nbsp; &nbsp; 我们先想下，比较理想的模型应该是什么样的，给定输入的图片矩阵，直接输出这张图片是什么，比如这张图片是狗，那就输出一个【dog】。这显然有点困难，毕竟从图片到文字人类进化了几百万年？我们简化一下问题，输入一个图片矩阵，输出一个矩阵（或者列向量）这个向量的n维代表n种类别，我们让这个模型不管怎么样，必须归一类（或者你把其中一类定成Null）。那么最简单的模型是什么样的呢？没错，就是输出列向量，是输入列向量乘上一个矩阵，也就是输出是输入的线性组合。
$$f(W,x_i) = W  x_i + b$$
![图示](https://upload-images.jianshu.io/upload_images/9592233-bb6e904ed93b52de.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

&nbsp; &nbsp; 我们希望通过学习，定出参数矩阵$W$。
### 3.3 线性回归-理解
&nbsp; &nbsp; 为什么这样可以分类呢。我们在一个二维平面上，给一堆点，我们可以使用一条直线把它分成两部分，这个时候这条直线就是$y = kx + b$我们与这个类似，线性回归，可以看作把这些图片映射到高维空间，在高维空间，他们是线性可分的。
![例子](https://upload-images.jianshu.io/upload_images/9592233-c6ab3752bc155dbf.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 4. KNN实现（Python）
### 4.1 环境
* Python 3.6.6
        * Numpy == 1.14.5
 * Pycharm 2018.2 Professional
 * DataSet
        * CIFAR-10 DataSet
### 4.2 代码
```Python
# 这是KNN分类器的示例程序
# 2018-8-15

import numpy as np

par_k = 3
#定义Knn的K
path = "E:\\2017-2018 Summer Vacation\\cs231n_program\\DataSets\\"
#这是存放数据集的路径，这个数据集我自己处理过
train_images = np.reshape(np.load(path + "cifar10_images.npy"), (10000, 3 * 32 * 32))
train_labels = np.load(path + "cifar10_labels.npy")
validation_images = np.reshape(np.load(path + "cifar10_images_validation.npy"), (10000, 3 * 32 * 32))
validation_labels = np.load(path + "cifar10_labels_validation.npy")

train_images = train_labels[:10]
train_labels = train_labels[:10]
validation_images = train_labels[:10]
validation_labels = train_labels[:10]
#不想使用整个数据集做实验的可以从中选一部分，要不然根本跑不完，
#i5-5200U下跑1000就费劲了
print(train_images.shape)
print(train_labels.shape)
print(validation_images.shape)
print(validation_labels.shape)

#L1范数
def distance_l1(a, b):
    return np.sum(np.abs(a - b), axis=1)

#L2范数
def distance_l2(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2)))

#定义一个正确率的计算
def accuracy(a, b):
    total = a.shape[0]
    counter = 0
    for i in range(0, total):
        if a[i] == b[i]:
            counter = counter + 1
    return counter / total

#正儿八经的代码
class KNN(object):
    def __init__(self):
        k = par_k
        pass
    #训练代码就是把数据存进去
    def train(self, train_im, train_la):
        self.tr_data = train_im
        self.tr_label = train_la
        pass
    #预测代码就是把每个距离算出来，依次比较
    def predict(self, input_vector):
        k = par_k
        round = input_vector.shape[0]
        n_train = self.tr_data.shape[0]
        dis_array = np.zeros((round, n_train))
        label_test = np.zeros(round)
        for i in range(0, round):
            for j in range(0, n_train):
                dis_array[i, j] = distance_l2(input_vector[i], self.tr_data[j])
        for i in range(0, round):
            temp = np.argsort(dis_array[i])
            temp_label = np.zeros(par_k)
            for j in range(0, par_k):
                temp_label[j] = self.tr_label[temp[j]]
            label_test[i] = np.argmax(np.bincount(temp_label.astype(int)))
        return label_test



if __name__ == "__main__":
    myknn = KNN()
    myknn.train(train_im=train_images, train_la=train_labels)
    predict_results = myknn.predict(input_vector=validation_images)
    accuracy = accuracy(predict_results, validation_labels)
    print(accuracy)

```
