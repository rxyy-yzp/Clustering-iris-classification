# 导入库及各模块
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import pygal

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

iris = pd.read_csv(r'./data/iris.csv',encoding = "gb18030")# 将data文件夹中iris.csv中数据存入iris
X = iris.iloc[:,:4]# 将种类一列去除，只取花萼长度、花萼宽度、花瓣长度、花瓣宽度4个维度的数据，并存入X

kmeans = KMeans(n_clusters = 3)# 构造聚类器，其中簇的个数取为3
kmeans.fit(X)# 聚类

X['聚类簇分布'] = kmeans.labels_# 获取聚类标签
centers = kmeans.cluster_centers_# 获取三个簇的簇中心

# 绘制横轴为花瓣长度，纵轴为花瓣宽度的聚类效果的散点图
sns.lmplot(x = '花瓣长度', y = '花瓣宽度', hue = '聚类簇分布', markers = ['s','^','o'],
           data = X, fit_reg = False, scatter_kws = {'alpha':0.8}, legend_out = False)
plt.scatter(centers[:,2], centers[:,3], marker = '*', color = 'black', s = 130)
plt.xlabel('花瓣长度')# 图中横轴名称
plt.ylabel('花瓣宽度')# 图中纵轴名称
plt.savefig("./data/花瓣长度与宽度聚类簇分布图.png",bbox_inches = 'tight')# 保存图片到data文件夹下

iris['原始种类分布'] = iris.种类.map({'virginica':0,'setosa':1,'versicolor':2})# 增加辅助列，将不同的花种映射到0,1,2三种值，方便原始图形与聚类的图形对比
# 绘制横轴为花瓣长度，纵轴为花瓣宽度的原始数据三个类别的散点图
sns.lmplot(x = '花瓣长度', y = '花瓣宽度', hue = '原始种类分布', data = iris, markers = ['s','^','o'],
           fit_reg = False, scatter_kws = {'alpha':0.8}, legend_out = False)
plt.xlabel('花瓣长度')# 图中横轴名称
plt.ylabel('花瓣宽度')# 图中纵轴名称
plt.savefig("./data/花瓣长度与宽度原始种类分布图.png",bbox_inches = 'tight')# 保存图片到data文件夹下

# 调用Radar这个类，并设置雷达图的填充，及数据范围
radar_chart = pygal.Radar(fill = True)
# 添加雷达图各顶点的名称
radar_chart.x_labels = ['花萼长度','花萼宽度','花瓣长度','花瓣宽度']
# 绘制三个雷达图区域，代表三个簇中心的指标值
radar_chart.add('C1', centers[0])
radar_chart.add('C2', centers[1])
radar_chart.add('C3', centers[2])
radar_chart.render_to_file('./data/雷达分析图.svg')# 保存图像


from sklearn.cluster import KMeans
KMeans(n_clusters=8,init='k-means++',n_init=10,max_iter=300, tol=0.0001,
       precompute_distances='auto',verbose=0,random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

