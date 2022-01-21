# 导入库及各模块
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

iris = pd.read_csv(r'./data/iris.csv',encoding = "gb18030")# 将data文件夹中iris.csv中数据存入iris
X = iris.iloc[:,:4]# 将种类一列去除，只取花萼长度、花萼宽度、花瓣长度、花瓣宽度4个维度的数据，并存入X

# 构造自定义函数，用于绘制不同k值和对应轮廓系数的折线图
def k_silhouette(X, clusters):
    K = range(2,clusters+1)
    S = []# 构建空列表，用于存储个中簇数下的轮廓系数
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        S.append(metrics.silhouette_score(X, labels, metric='euclidean'))# 调用字模块metrics中的silhouette_score函数，计算轮廓系数

    plt.style.use('ggplot')# 设置绘图风格
    plt.plot(K, S, 'b*-')# 绘制K的个数与轮廓系数的关系
    plt.xlabel('簇的个数')# 图中横轴名称
    plt.ylabel('轮廓系数')# 图中纵轴名称
    plt.savefig("./data/轮廓系数法确定K值.png", bbox_inches='tight')# 保存图片到data文件夹下
    plt.show()# 展现图片

# 自定义函数的调用
k_silhouette(X, 15)
