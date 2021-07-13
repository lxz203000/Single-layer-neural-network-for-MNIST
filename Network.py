import numpy as np
import matplotlib.pyplot as plt
import math

class Network(object):

    def __init__(self, hidden_size, input_size = 256, output_size = 10, std = 1e-4):
        
        self.params = {}
        
        #正态分布作为初始值
        W1=np.random.normal(loc=0,scale=std,size=input_size*hidden_size)
        W1=W1.reshape(input_size,hidden_size)
        W2=np.random.normal(loc=0,scale=std,size=hidden_size*output_size)
        W2=W2.reshape(hidden_size,output_size)

        self.params['W1'] = W1
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = W2
        self.params['b2'] = np.zeros(output_size)
        
        return
    
    #sofrmax函数
    def softmax(self,x):
        tmp = np.max(x,axis=1) #得到每行的最大值，避免溢出
        x -= tmp.reshape((x.shape[0],1)) #缩放元素
        x = np.exp(x) #计算所有值的指数
        tmp = np.sum(x, axis = 1) #每行求和        
        x /= tmp.reshape((x.shape[0], 1)) #求softmax
        return x

    #交叉熵函数
    def cross_entropy(self,x,y):
        delta=1e-8 #添加一个微小值可以防止负无限大(np.log(0))的发生。
        return -np.sum(x*np.log(y+delta))

    def forward_pass(self, X, y = None, wd_decay = 0.0):
    
        loss = None
        predict = None
        W1=self.params['W1']
        W2=self.params['W2']
        #求隐层、输出层和预测值
        self.h=np.maximum(0,np.dot(X,W1)+self.params['b1'])
        #self.c=np.maximum(0,np.dot(self.h,W2)+self.params['b2'])
        #如果使用ReLU，因为第一次采用正态分布，所以c均为0，则数值计算梯度无法完成
        self.c=np.dot(self.h,W2)+self.params['b2']
        predict=np.argmax(self.c[0])
        for i in range(1,len(self.c)):
            predict=np.append(predict,np.argmax(self.c[i]))
        
        if y is None:
            #返回预测值
            return predict
        else:
            #返回loss
            a=self.softmax(self.c) #对输出层求softmax
            #构建目标的概率
            y1=np.zeros_like(a)
            y1[range(len(a)),list(y)]=1
            #对每组数据求loss均值
            loss=self.cross_entropy(y1[0],a[0])+wd_decay*(np.sum(np.square(W1)) + np.sum(np.square(W2)))/2
            for i in range(1,len(y1)):
                loss=np.append(loss,self.cross_entropy(y1[i],a[i])+wd_decay*(np.sum(np.square(W1)) + np.sum(np.square(W2)))/2)
            loss=np.sum(loss)/len(loss)
            return loss
        

    def back_prop(self, X, y, wd_decay = 0.0):
        grads = {}
        W1=self.params['W1']
        W2=self.params['W2']
        
        #对W2和b2的梯度
        err=self.softmax(self.c)
        err[range(len(X)),list(y)]-=1
        err/=len(X)
        grads['W2']=self.h.T.dot(err)+wd_decay*W2
        grads['b2']=np.sum(err,axis=0)
        #对W1和b1的梯度
        err2=err.dot(W2.T)
        err2=(self.h>0)*err2
        grads['W1']=X.T.dot(err2)+wd_decay*W1
        grads['b1']=np.sum(err2,axis=0)
        
        return grads
 
    def numerical_gradient(self, X, y, wd_decay = 0.0, delta = 1e-6):
        grads = {}
            
        for param_name in self.params:
            grads[param_name] = np.zeros(self.params[param_name].shape)
            itx = np.nditer(self.params[param_name], flags=['multi_index'], op_flags=['readwrite'])
            while not itx.finished:
                idx = itx.multi_index
                #This part will iterate for every params
                #You can use self.parmas[param_name][idx] and grads[param_name][idx] to access or modify params and grads
                #
                # TODO
                #
                self.params[param_name][idx]+=delta
                L1=self.forward_pass(X,y,wd_decay)
                self.params[param_name][idx]-=2*delta
                L2=self.forward_pass(X,y,wd_decay)
                grads[param_name][idx]=(L1-L2)/(2*delta)
                self.params[param_name][idx]+=delta
                itx.iternext()
        return grads
    
    def get_acc(self, X, y):
        pred = self.forward_pass(X)
        return np.mean(pred == y)
    
    def train(self, X, y, X_val, y_val,
                learning_rate=0, 
                momentum=0, do_early_stopping=False, alpha = 0,
                wd_decay=0, num_iters=10,
                batch_size=4, verbose=False, print_every=10,do_learning_rate_decay=False,learning_rate_decay=0.98):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        acc_history = []
        val_acc_history = []
        val_loss_history = []
        #early stopping所需参数
        best_val_loss=1e8
        best_params=self.params

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #learning rate decay
            if do_learning_rate_decay:
                decay_learning_rate = learning_rate * np.power(learning_rate_decay,(it // 200))
            else:
                decay_learning_rate=learning_rate
            #随机选取batch
            idx_batch=np.random.choice(np.arange(num_train),size=batch_size)
            X_batch=X[idx_batch]
            y_batch=y[idx_batch]
            #算出loss和grads
            loss=self.forward_pass(X=X_batch,y=y_batch,wd_decay=wd_decay)
            grads=self.back_prop(X=X_batch,y=y_batch,wd_decay=wd_decay)
            val_loss=self.forward_pass(X=X_val,y=y_val,wd_decay=wd_decay)
            loss_history.append(loss)
            val_loss_history.append(val_loss)
            #梯度下降，使用momentum，并更新
            v1=np.zeros_like(self.params['W1'])
            v2=np.zeros_like(self.params['W2'])
            vb1=np.zeros_like(self.params['b1'])
            vb2=np.zeros_like(self.params['b2'])

            v1=momentum*v1-decay_learning_rate*grads['W1']
            v2=momentum*v2-decay_learning_rate*grads['W2']
            vb1=momentum*vb1-decay_learning_rate*grads['b1']
            vb2=momentum*vb2-decay_learning_rate*grads['b2']

            self.params['W1']+=v1
            self.params['W2']+=v2
            self.params['b1']+=np.ravel(vb1)
            self.params['b2']+=np.ravel(vb2)

            
            if verbose and it % print_every == 0:
                print('iteration %d / %d: training loss %f val loss: %f' % (it, num_iters, loss, val_loss))
 
            if it % iterations_per_epoch == 0:
                
                train_acc = self.get_acc(X_batch, y_batch)
                val_acc = self.get_acc(X_val, y_val)
                acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
            if do_early_stopping:
                pass
                #
                # TODO: early stopping
                #
                if val_loss<best_val_loss:
                    best_val_loss=val_loss
                    best_params=self.params
                #计算度量进展，其中8也可作为超参数
                P=(np.sum(val_loss_history[-8:])/(8*min(val_loss_history[-8:]))-1)*1000
                #计算泛化损失
                GL=100*(val_loss/best_val_loss-1)
                #大于alpha则停止
                if (GL/P)>alpha:
                    self.params=best_params.copy()
                    break

        return {
          'loss_history': loss_history,
          'val_loss_history': val_loss_history,
          'acc_history': acc_history,
          'val_acc_history': val_acc_history,
        }