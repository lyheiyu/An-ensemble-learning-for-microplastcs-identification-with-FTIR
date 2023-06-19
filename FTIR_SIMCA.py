from simca import  SIMCA
from sklearn.model_selection import  train_test_split
from FTIR_HCA import HCA

import numpy as np
import pandas as pd
from plsda import  PLSDA
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90,size=15)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name,size=15)    # 将标签印在y轴坐标上
    plt.ylabel('True label',font2)
    plt.xlabel('Predicted label',font2)
    plt.show()
def parseData2(fileName, begin, end):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

    polymerName = dataset.iloc[1:, 1]

    waveLength = dataset.iloc[0, begin:end]
    intensity = dataset.iloc[1:, begin:end]
    polymerName = np.array(polymerName)
    # waveLength = np.array(waveLength)

    polymerID = dataset.iloc[1:, 1764]
    polymerID = np.array(polymerID)
    intensity = np.array(intensity)
    x_class=[]
    y_class=[]
    for i in range(12):
        m=[]
        z=[]
        for j in range(len(intensity)):
            if int(polymerID[j])==i+1:

                z.append(polymerID[j])
                m.append(intensity[j])
        x_class.append(m)
        y_class.append(z)

    return polymerName, waveLength, intensity, polymerID,x_class,y_class
polymerName, waveLength, intensity, polymerID,x_each,y_each = parseData2('D4_4_publication5.csv', 2, 1763)
#x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
matrixC=[]
for i in range(12):
    print(len(x_each[i]))
    #x_train, x_test, y_train, y_test = train_test_split(x_each[i], y_each[i], test_size=0.3, random_state=1)
    sc = SIMCA(n_components=3, alpha=0.05)
    _ = sc.fit(x_each[i], y_each[i])
    xe = []
    for i in range(12):
        pred = sc.predict(x_each[i])
        x=0
        for item in pred:

            if str(item)=='True':
                x+=1
            #print(np.array(pred).shape[0])
        print(str(i+1)+'class:'+'accuracy：'+str(x/np.array(pred).shape[0]))
        xe.append(x)
    matrixC.append(xe)
PN=[]
for item in polymerName:
    if item not in PN:
        PN.append(item)
print(PN)
data1 = pd.DataFrame(PN)

matrixC=np.array(matrixC)
matrixC=pd.DataFrame(matrixC)
matrixC.to_csv('SIMCA5.csv')
plot_confusion_matrix(matrixC,PN,'SIMCA')
print(matrixC.shape)
# sc=SIMCA(n_components=30,alpha=0.05)
# sp = PLSDA(n_components=30, style='soft')
# x_train=SIMCA.matrix_X_(SIMCA,x_train)
# x_test=SIMCA.matrix_X_(SIMCA,x_test)
# y_train=SIMCA.column_y_(SIMCA,y_train)
# y_test=SIMCA.column_y_(SIMCA,y_test)
# _ = sc.fit(x_train, y_train)
# pred=sc.predict(x_test)
# print(pred)
# #pred=sc.predict(x_train)
# simcaScore= sc.score(x_test,pred.ravel())
# print(simcaScore)