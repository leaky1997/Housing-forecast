# -*- coding: utf-8 -*-
"""
Created on Fri May 15 08:32:20 2020

@author: 李奇
"""

from math import sqrt
from sklearn import svm
import scipy.io as io
from os.path import splitext
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



def read_file(f, logging=False):
    '''
    读取csv文件
    '''
    datatype=splitext(f)[1]

    
    
    
    print("==========读取数据=========")
    if datatype =='.csv':
        data =  pd.read_csv(f,encoding='gbk')                                      #gbk
    if datatype =='.xlsx':
        data =  pd.read_excel(f,encoding='gbk')
    if datatype =='.xls':
        data =  pd.read_excel(f,encoding='gbk')
        
    if logging:  
        print(data.head(5))  
        print(f, "包含以下列")  
        print(data.columns.values)  
        print(data.describe())  
        print(data.info()) 
    print("==========读取完毕=========")
    return data  


def transfer_to_np1(filename = 'soochow456.csv',scaler=False):
    '''
    将dataframe转换成array格式并归一化
    '''
    print("==========数据预处理=======")
    data = read_file(filename)
    data = data.dropna()
    datapart = data.iloc[:,:37]

    x = np.array(datapart)
    x = x[1:]
    y = data.iloc[:,-1]
    y = y[1:]
    y = np.array(y)
    y=y.reshape(-1,1)
    
    x.astype('float64')
    y.astype('float64')
    
    if scaler:
        x_scaler = MinMaxScaler(feature_range=(0, 1))###不归一化
        x = x_scaler.fit_transform(x)

        y_scaler = MinMaxScaler(feature_range=(0, 1))###不归一化
        y = y_scaler.fit_transform(y)
        
        
        print("==========预处理完毕=======")
    
        return x,y,x_scaler,y_scaler
    
    print("==========预处理完毕=======")
    
    return x,y




def datasets_creation(x,y,test_size=0.2):
    x_train,x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   test_size = test_size,
                                                   random_state = 0)
    return x_train,x_test, y_train, y_test
def regression(X,Y,method='svm'):
    '''
    分类器
    '''
    print("=======开始训练分类器======")
    print('采用的分类器为',method)
    if method=='svm':
        
        clf = svm.SVR(gamma='auto')

    # 方法选择
    # 1.决策树回归
    if method == 'tree':
        from sklearn import tree
        clf = tree.DecisionTreeRegressor()
        

        # 2.线性回归
    if method == 'linear' :
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()

         
        # 3.SVM回归

         
        # 4.kNN回归
    if method == 'knn':
        from sklearn import neighbors
        clf = neighbors.KNeighborsRegressor()
         
        # 5.随机森林回归
    if method == 'RFR':
        from sklearn import ensemble
        clf = ensemble.RandomForestRegressor(n_estimators=20)  # 使用20个决策树
    if method == 'Adaboost':
        # 6.Adaboost回归
        from sklearn import ensemble
        clf = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
    if method == 'GBR':
        # 7.GBRT回归
        from sklearn import ensemble
        clf = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
    if method == 'Bag':
        # 8.Bagging回归
        from sklearn import ensemble
        clf = ensemble.BaggingRegressor()
    if method == 'ETR':
        # 9.ExtraTree极端随机数回归
        from sklearn.tree import ExtraTreeRegressor
        clf = ExtraTreeRegressor()      
        
    if method == 'MLP':
        from sklearn.neural_network import MLPRegressor
        clf = MLPRegressor(solver='adam',alpha=1e-5, hidden_layer_sizes=(100,4), random_state=1)
    

        
        
    clf.fit(X, Y)
        
    print("==========训练完毕=========")
    
    
    
    
    
    return clf


from keras import models
from keras import layers
 
def build_model(x):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    #input_shape为传入一个shape给第一层，为13行的矩阵,所以要求输入的数据为13列的矩阵
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(x.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

def score(y_pred, y_test):
    
    
    rmse = sqrt(mean_squared_error(y_pred, y_test))
    
    return rmse

def plot(y_test,y_pred,name):
    import matplotlib.pyplot as plt
    
    plt.plot(y_test, label='actual')
    
    plt.plot(y_pred, label='forecast')
    plt.legend()
    plt.savefig(name+'predition.png',dpi=512)
    plt.show()
#%%
if __name__=='__main__':
    dataset1='外部评价计算权重2--数据进行了更新.xlsx'
    dataset2='扬州小区的数据处理（37个输入和1个输出）.xls'
#    for method in ['svm','tree','linear','knn','RFR','Adaboost','GBR','Bag','ETR','MLP']:
    method= 'ANN'
        
    x, y, x_scaler,y_scaler= transfer_to_np1(dataset2,scaler=True)
    
    y = y.reshape(-1)
    
#    x_train,x_test, y_train, y_test = datasets_creation(x,y,test_size=0.1)
    
    k = 4
    num_val_samples = len(x) // k
    num_epochs = 100
    all_scores = []
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
        #准备验证数据：第k个分区的数据
        val_data = x[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
     
        #准备训练数据：其他所有分区的数据
        partial_train_data = np.concatenate(
            [x[:i * num_val_samples],
             x[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y[:i * num_val_samples],
             y[(i + 1) * num_val_samples:]],
            axis=0)
     
        # 构建已编译的模型
        model = build_model(x)
        #训练模型，为静默模式，即不在标准输出流输出日志信息(verbose=0),训练100轮
        history = model.fit(partial_train_data, partial_train_targets,
                  epochs=num_epochs, batch_size=16, verbose=1)
        
        
        mae_history = history.history['mean_absolute_error']
        all_mae_histories.append(mae_history)         
        #在验证数据集上评估模型
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
        all_scores.append(val_mae)

    
#    clf = regression(x_train,y_train.reshape(-1,),method=method)
#    
#    
        y_pred = model.predict(val_data)
        
        y_pred = np.array(y_pred)
        score1 = score(y_pred, val_targets)
        
        y_pred_real = y_scaler.inverse_transform(y_pred.reshape(-1,1))
        y_test_real = y_scaler.inverse_transform(val_targets.reshape(-1,1))
        score2 = score(y_pred_real, y_test_real)
        
        
        plot(val_targets,y_pred,str(score1)+method+str(k))
        plot(y_test_real,y_pred_real,str(score2)+method+str(k))
    
    
    