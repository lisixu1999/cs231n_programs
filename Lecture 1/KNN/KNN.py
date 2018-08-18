# 这是KNN分类器的示例程序
# 2018-8-15

import numpy as np

par_k = 3
# 定义Knn的K
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
# 不想使用整个数据集做实验的可以从中选一部分，要不然根本跑不完，
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
