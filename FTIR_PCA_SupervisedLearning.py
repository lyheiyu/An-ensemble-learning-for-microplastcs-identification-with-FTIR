import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import sklearn.preprocessing as sp
MMScaler = MinMaxScaler()
def mkdir(path):
    # 判断目录是否存在
    # 存在：True
    # 不存在：False
    folder = os.path.exists(path)

    # 判断结果
    if not folder:
        # 如果不存在，则创建新目录
        os.makedirs(path)
        print('-----创建成功-----')

    else:
        # 如果目录已存在，则不创建，提示目录已存在
        print(path + '目录已存在')


def parseData2(fileName, begin, end):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

    polymerName = dataset.iloc[1:971, 1]
    waveLength = dataset.iloc[0, begin:end]
    intensity = dataset.iloc[1:971, begin:end]
    polymerName = np.array(polymerName)
    waveLength = np.array(waveLength)

    polymerID = dataset.iloc[1:, 1764]
    polymerID = np.array(polymerID)

    intensity = np.array(intensity)
    return polymerName, waveLength, intensity, polymerID
#data_x,id=generationData(30,35,1)

#计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征
def meanX(dataX):
    return np.mean(dataX,axis=0)#axis=0表示按照列来求均值，如果输入list,则axis=1
def pca(XMat, k):
    average = meanX(XMat)

    m,n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        #注意特征向量时列向量
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里需要进行转置，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        print( '############',selectVec[0].reshape(1,-1))
        PC1=selectVec[0].reshape(1761)
        print(PC1.shape)
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData,selectVec
def getPN(fileName):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

    polymerName = dataset.iloc[1:, 1]
    PN = []
    for item in polymerName:
        if item not in PN:
            PN.append(item)
    return PN
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90,size=15)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name,size=15)  # 将标签印在y轴坐标上
    plt.ylabel('True label',font2)
    plt.xlabel('Predicted label',font2)
    plt.show()
def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    #dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        #axis_x2.append(dataArr2[i,0])
        #axis_y2.append(dataArr2[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
   # ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    #plt.savefig("outfile.png")
    plt.show()
def main(i):
    datafile = "data.txt"
    polymerName, waveLength, intensity, polymerID = parseData2('D4_4_publication5.csv', 2, 1763)

    k = (i+1)
    return pca(intensity, k)
if __name__ == "__main__":
    PCAscoreList=[]
    for i in range(103,104):
        PCAscore=[]
        finalData, reconMat ,Vector= main(i)

        finalData=np.array(finalData,dtype=np.float32)
        print(finalData.shape)

        # ohe = sp.OneHotEncoder(sparse=False, dtype="int32")
        # result = ohe.fit_transform(finalData)
        #finalData = MMScaler.fit_transform(finalData)
        #modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,128), random_state=1)
        polymerName, waveLength, intensity, polymerID = parseData2('D4_4_publication5.csv', 2, 1763)
        x_train, x_test, y_train, y_test = train_test_split(finalData, polymerID, test_size=0.3, random_state=0)
        x_train = np.array(x_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)

        modelRF = RandomForestClassifier(n_estimators= 74,oob_score=True,
                                     random_state=10,max_depth=11,
                                    min_samples_split=2,min_samples_leaf=2,max_features=9)
        modelLDA = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                              solver='svd', store_covariance=False, tol=0.001)
        modelSVM = svm.SVC(C=1, kernel='linear', decision_function_shape='ovo')
        modelKNN = KNeighborsClassifier(n_neighbors=5)
        # modelRF= RandomForestClassifier(n_jobs=-1)
        # modelRF.fit(x_train,y_train)
        modelMLP.fit(x_train, y_train)
        modelLDA.fit(x_train, y_train)
        modelKNN.fit(x_train, y_train)
        modelSVM.fit(x_train, y_train)
        modelRF.fit(x_train,y_train)
        print('PCA-MLP:' + str(i  + 1) + ':' + str(modelMLP.score(x_test, y_test)))
        print('PCA-SVM:' + str(i  + 1) + ':' + str(modelSVM.score(x_test, y_test)))
        print('PCA-KNN:' + str(i  + 1) + ':' + str(modelKNN.score(x_test, y_test)))
        print('PCA-LDA:' + str(i  + 1) + ':' + str(modelLDA.score(x_test, y_test)))
        print('PCA-RF:' + str(i  + 1) + ':' + str(modelRF.score(x_test, y_test)))
        PCAscore.append(modelSVM.score(x_test, y_test))
        PCAscore.append(modelKNN.score(x_test, y_test))
        PCAscore.append(modelLDA.score(x_test, y_test))
        PCAscore.append(modelRF.score(x_test, y_test))
        PCAscoreList.append(PCAscore)
        # print('UMAP-RF:' +str(i*10+10)+':'+ str(modelRF.score(x_test, y_test)))
    PCA_supervise_report=pd.DataFrame(PCAscoreList)
    PCA_supervise_report.to_csv('PCA+RF_supervise_report.csv')