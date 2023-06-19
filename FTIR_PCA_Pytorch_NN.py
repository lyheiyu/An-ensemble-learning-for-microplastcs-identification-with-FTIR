import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
MMScaler = MinMaxScaler()
def parseData2(fileName, begin, end):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
    polymerName = dataset.iloc[1:971, 1]
    waveLength = dataset.iloc[0, begin:end]
    intensity = dataset.iloc[1:971, begin:end]
    polymerName = np.array(polymerName)
    waveLength = np.array(waveLength)
    # polymerNameList = {}.fromkeys(polymerName).keys()
    polymerNameList = []
    pList = list(set(polymerName))
    for item in pList:
        polymerNameList.append(item)
    polymerNameList = np.array(polymerNameList)
    polymerNameID = []
    for i in range(len(polymerNameList)):
        polymerNameID.append(i + 1)
    polymerNameData = []
    for item in polymerName:
        for i in range(len(pList)):
            if item == pList[i]:
                polymerNameData.append(i)

    # x_train = normalize(x_train, norm='l2')
    # x_test = normalize(x_test, norm='l2')


    intensity = np.array(intensity)

    for item in intensity:

        for i in range(len(item)):
            if item[i] == '':
                item[i] = 0
    #intensity = MMScaler.fit_transform(intensity)
    return polymerName, waveLength, intensity, polymerNameData
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
def main():
    datafile = "data.txt"
    polymerName, waveLength, intensity, polymerID = parseData2('D4_4_publication5.csv', 2, 1763)

    k = 300
    return pca(intensity, k)
if __name__ == '__main__':
    finalData, reconMat, Vector = main()
    batch_size=290
    LR = 0.005  # learning rate
    EPOCH = 10
    finalData = np.array(finalData)

    # finalData = MMScaler.fit_transform(finalData)
    #modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1)
    polymerName, waveLength, intensity, polymerID = parseData2('D4_4_publication5.csv', 2, 1763)
    x_train, x_test, y_train, y_test = train_test_split(finalData, polymerID, test_size=0.3, random_state=1)
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    x_train, x_test, y_train, y_test = map(torch.tensor, (x_train, x_test, y_train, y_test))
    y_train = y_train.long()
    y_test = y_test.long()
    train_dataset = TensorDataset(x_train, y_train)

    test_dataset = TensorDataset(x_test, y_test)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = nn.Linear(300, 128)
            self.l2 = nn.Linear(128, 128)
            self.l3 = nn.Linear(128, 64)
            self.l4 = nn.Linear(64, 64)
            self.l5 = nn.Linear(64, 64)
            self.l6 = nn.Linear(64, 32)
            self.l7 = nn.Linear(32, 32)
            self.l8 = nn.Linear(32, 12)

        def forward(self, x):
            # Flatten the data (n, 1, 28, 28) --> (n, 784)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            x = F.relu(self.l4(x))
            x = F.relu(self.l5(x))
            x = F.relu(self.l6(x))
            x = F.relu(self.l7(x))
            return F.log_softmax(self.l8(x), dim=1)
            # return self.l5(x)


    for i in range(1):
        model = Net()
        Accuracy = []
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


        def train(epoch):
            # 每次输入barch_idx个数据
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                output = model(data)
                # loss
                loss = F.nll_loss(output, target)
                loss.backward()
                # update
                optimizer.step()
                if batch_idx % 200 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data))


        def test():
            test_loss = 0
            correct = 0
            # 测试集
            index=0

            #result=Variable()
            for data, target in test_loader:

                index+=1
                data, target = Variable(data, volatile=True), Variable(target)
                print(len(data))
                output = model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target).data
                # get the index of the max
                pred = output.data.max(1, keepdim=True)[1]
                result=pred
                Ptrue=target
                print('precision:'+str(precision_score(target,pred)),index)
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            if epoch % 999 == 0:

                PN = getPN('D4_4_publication5.csv')
                # t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
                # SVM_report = pd.DataFrame(t)
                # SVM_report.to_csv('MLP_report5.csv')
                MLPcm =confusion_matrix(Ptrue,result)
                plot_confusion_matrix(MLPcm, PN, 'PCA-PytrochNN')
                Accuracy.append(100. * correct / len(test_loader.dataset))


        for epoch in range(1, 1000):
            train(epoch)
            test()