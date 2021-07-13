# Single-layer-neural-network-for-MNIST
MNIST是视觉处理方面的结构比较简单的标准数据集。包括手写16\*16像素的黑白手写数字图像和对应的数字标记，并分为train,validation和test三组。图像数据已经转换为256长度的灰度值向量，标签的值为0-9。training set用于训练网络，validation set用于在训练过程中观察训练是否出现了过拟合。test set用于训练结束后评价训练结果。

### 其他文件包括：
* X_test.npy X_train.npy X_val.npy Y_test.npy Y_train.npy Y_val.npy 数据文件，已经过整理
* test_utils.py test_pred.npy 帮助检验计算正确性的模组

### 数据格式：
* X: 为np.array，形状为\[N,256\]，其中N为数据的数量。
* y: 为np.array，形状为\[N,\]，N为数据的数量。y的值为0-9，对应相应的标签。
