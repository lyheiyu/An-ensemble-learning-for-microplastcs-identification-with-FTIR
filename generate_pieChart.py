import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import  train_test_split
class FTIR_PieChart:
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
        x_class = []
        y_class = []
        for i in range(12):
            m = []
            z = []
            for j in range(len(intensity)):
                if int(polymerID[j]) == i + 1:
                    print(i)
                    z.append(polymerID[j])
                    m.append(intensity[j])
            x_class.append(m)
            y_class.append(z)

        return polymerName, waveLength, intensity, polymerID, x_class, y_class
if __name__ == '__main__':
    FTIR_Pie=FTIR_PieChart
    polymerName, waveLength, intensity, polymerID, x_each, y_each =FTIR_Pie.parseData2('D4_4_publication6.csv', 2, 1763)
    polymerNameSet=[]
    #polymerIdSet=[]
    for polymerN in polymerName:
        if polymerN not in polymerNameSet:
            polymerNameSet.append(polymerN)
    polymerNameSet=np.array(polymerNameSet)

    percent2=np.zeros(len(polymerNameSet))
    for pName in polymerName:
        for index in range(len(polymerNameSet)):
            if pName == polymerNameSet[index]:
                percent2[index] += 1


    # dataFinal=data['Tweets'].append(data['SA'])
    # for item in dataFinal:
    #     print(item['SA'])
    # Pie chart:
    # pie_chart = pd.Series(percent, index=sources, name='Sources')
    # pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6))
    #colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#504ea3"]
    colors = ['c', 'b', 'g', 'r', 'm', 'y', '#377eb8', 'darkviolet', 'olive',
              'tan', 'lightgreen', 'gold', 'cyan', 'magenta', 'pink',
              'crimson', 'navy', 'cadetblue']

    #soldNums = [0.05,0.45,0.15,0.35]
    plt.cla()
    # 绘制饼状图
    plt.pie(percent2,
            labels=polymerNameSet,
            autopct="%3.1f%%",
            startangle=60,
            colors=colors)

    plt.title("Modified data")
    plt.clf()
    plt.ylim(-70, 110)
    plt.xlim(4000,500)
    for j in range(len(intensity)):
        for i in range(len(polymerNameSet)):
            if polymerID[j]==str(i+1):
                plt.plot(waveLength,intensity[j],color=colors[i])
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')

    plt.title('Modified data set')
    #plt.savefig('Modified.png',bbox_inches='tight')
    plt.show()