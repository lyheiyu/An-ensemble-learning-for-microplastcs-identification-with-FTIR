import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import isnan
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  svm
from sklearn.model_selection import train_test_split
class TestSVM:
    def __init__(self):
        pass
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
        plt.xticks(num_local, labels_name, rotation=90, size=15)  # 将标签印在x轴坐标上
        plt.yticks(num_local, labels_name, size=15)  # 将标签印在y轴坐标上
        plt.ylabel('True label', font2)
        plt.xlabel('Predicted label', font2)
        plt.show()

    def parseData2(fileName,begin,end):

        dataset = pd.read_csv(fileName, header=None, encoding='latin-1',keep_default_na=False)
        polymerName=dataset.iloc[1:971,1]
        waveLength=dataset.iloc[0,begin:end]
        intensity=dataset.iloc[1:971,begin:end]
        polymerName= np.array(polymerName)
        waveLength=np.array(waveLength)
        #polymerNameList = {}.fromkeys(polymerName).keys()
        polymerNameList=[]
        pList=list(set(polymerName))
        for item in pList:
            polymerNameList.append(item)
        polymerNameList=np.array(polymerNameList)
        polymerNameID=[]
        for i in range(len(polymerNameList)):
            polymerNameID.append(i+1)
        polymerNameData=[]
        for item in polymerName:
            for i in range(len(pList)):
                if item==pList[i]:
                    polymerNameData.append(i+1)


        intensity =np.array(intensity)
        for item in intensity:

            for i in range(len(item)):
                if item[i] == '':
                    item[i] = 0
        return polymerName,waveLength,intensity,polymerNameData
    def spectrumSVM(x,y,cvalue,kModel,decFunction):
        model=svm.SVC(C=cvalue, kernel=kModel, decision_function_shape=decFunction)

        model.fit(x,y)

        return model
if __name__ == '__main__':

    tS=TestSVM
    #this is for the 216_2018.csv
    #polymerName, polymerID, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM.csv',3,1179)
    #this is for the D4 public_csv
    polymerName, waveLength, intensity,polymerID= tS.parseData2('D4_4_publication5.csv', 2, 1763)
    pList = list(set(polymerName))
    print(max(polymerID))
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerName, test_size=0.3, random_state=1)
    accuracies=[]
    index=[]
    recallScore=[]
    model=tS.spectrumSVM(x_train,y_train,1,'linear','ovo')
    y_pre=model.predict(x_test)

    print(y_pre)


    PN = tS.getPN('D4_4_publication5.csv')
    t = classification_report(y_test, y_pre, target_names=PN, output_dict=True)
    SVM_report = pd.DataFrame(t)
    SVM_report.to_csv('SVM_report5.csv')
    cm=confusion_matrix(y_test,y_pre)
    KNN_Confusion_matrix=pd.DataFrame(cm)
    KNN_Confusion_matrix.to_csv('SVM_CM5.csv')

    tS.plot_confusion_matrix(cm,PN,'SVM')

#    model = tS.spectrumSVM(x_train, y_train)

    y_predict = model.predict(x_test)
        #print(y_predict)
    sum = len(y_predict)
    correct = 0
    for j in range(len(y_predict)):
        if y_predict[j] == y_test[j]:
            correct = correct + 1
    accuracies.append(correct/sum)
    confusion_matrix(y_test,y_predict,labels=pList)
    recallScore.append(recall_score(y_test, y_predict, labels=None, pos_label=1, average='micro',
sample_weight=None))
        #print(index)
    print('score:'+str(model.score(x_test,y_test)))
    print('accuracy:' + str(correct / sum))
    print('recall accuracy:' + str(recallScore))

    #print(max(accuracies))
    #for i in range(len(accuracies)):
    # for i in range(len(accuracies)):
    #     if accuracies[i]==max(accuracies):
    #         print(accuracies[i])
    #         #print('maxValue：'+str(index[i]))
    #
    # fig, ax = plt.subplots()
    # ax.set_xlabel('K value')
    # ax.set_ylabel('accuracy')
    # ax.scatter(1,accuracies, 'b')
    # plt.show()
    # ax.legend()
    # plt.show()
    ####
    # accuracies2 = []
    # index2 = []
    # for i in range(3,40):
    #     model = tk.spectrumKNN(x_train, y_train, i)
    #     index2.append(i)
    #     y_predict2 = model.predict(x_test)
    #     #print(y_predict)
    #     sum = len(y_predict2)
    #     correct = 0
    #     for j in range(len(y_predict2)):
    #         if y_predict2[j] == y_test[j]:
    #             correct = correct + 1
    #     accuracies2.append(correct/sum)
    #
    #     #print(index)
    #     print('accuracy2:' + str(correct / sum))
    #
    # #print(max(accuracies))
    # #for i in range(len(accuracies)):
    # for i in range(len(accuracies2)):
    #     if accuracies2[i]==max(accuracies):
    #         print(accuracies2[i])
    #         print('maxValue2：'+str(index[i]))
    # #fig, ax = plt.subplots()
    # ax.set_xlabel('K value')
    # ax.set_ylabel('accuracy')
    # ax.plot(index2,accuracies2, 'r', label='KNN', linestyle='dotted')
    # #ax.legend()
    # ax.legend()
    # plt.show()
    #scores = cross_val_score(model, x_train, y_train, cv=20, scoring='accuracy')
    #n_scores.append(scores.mean())
    #predict=model.predict([intensity[3]])
    #print(polymerName)

    #print(scores)
    # print(model.score(x_test, y_test))
    # print(model.predict(x_test))
