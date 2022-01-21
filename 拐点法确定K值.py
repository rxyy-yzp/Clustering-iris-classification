# 导入库及各模块
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

iris = pd.read_csv(r'./data/iris.csv',encoding = "gb18030")# 将data文件夹中iris.csv中数据存入iris
X = iris.iloc[:,:4]# 将种类一列去除，只取花萼长度、花萼宽度、花瓣长度、花瓣宽度4个维度的数据，并存入X

# 构造自定义函数，用于绘制不同k值和对应总的簇内离差平方和的折线图
def k_SSE(X, clusters):
    K = range(1,clusters + 1)# 选择连续的K种不同的值
    TSSE = []# 构建空列表用于存储总的簇内离差平方和
    for k in K:
        SSE = []# 用于存储各个簇内离差平方和
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        labels = kmeans.labels_# 返回簇标签
        centers = kmeans.cluster_centers_# 返回簇中心
        # 计算各簇样本的离差平方和，并保存到列表中
        for label in set(labels):
            SSE.append(np.sum((X.loc[labels == label,] - centers[label,:]) ** 2))
        TSSE.append(np.sum(SSE))# 计算总的簇内离差平方和

    plt.style.use('ggplot')# 设置绘图风格
    plt.plot(K, TSSE, 'b*-')# 绘制K的个数与TSSE的关系折线图
    plt.xlabel('簇的个数')# 图中横轴名称
    plt.ylabel('簇内离差平方和之和')# 图中纵轴名称
    plt.savefig("./data/拐点法确定K值.png",bbox_inches = 'tight')# 保存图片到data文件夹下
    plt.show()# 展现图片

# 自定义函数的调用
k_SSE(X, 15)