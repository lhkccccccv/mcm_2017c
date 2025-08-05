import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import itertools
from enum import Enum
from . import get_data

plt.rcParams["font.family"] = ["SimHei"]
data1 = get_data.read_csv1()
colors = itertools.cycle(
    ["lime", "fuchsia", "red" "blue", "cyan", "yellow", "green", "purple"]
)
markers = itertools.cycle(["+", "x", "d"])


def scatter_coef(data):
    def scatter_cfg():
        plt.xlabel("浓度")
        plt.ylabel("颜色")
        plt.gca().xaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )  # x轴保留1位小数
        plt.gca().yaxis.set_major_formatter(
            FormatStrFormatter("%.2f")
        )  # y轴保留2位小数

    for i in data:
        x = []
        y = []
        if i == "组胺":
            group1 = data[i].iloc[:5, :]
            group2 = data[i].iloc[5:, :]
            group1 = group1.sort_values(by="ppm")
            group2 = group2.sort_values(by="ppm")
            x1, x2 = group1.iloc[:, 0], group2.iloc[:, 0]
            y1, y2 = group1.iloc[:, 1:], group1.iloc[:, 1:]
            for k in range(y1.shape[1]):
                plt.plot(x1, y1.iloc[:, k], "-o", markersize=4)
            scatter_cfg()
            plt.title(f"{i}的浓度与颜色的散点图1")
            plt.show()
            for j in range(y2.shape[1]):
                plt.plot(x2, y2.iloc[:, j], "-o", markersize=4)
            scatter_cfg()
            plt.title(f"{i}的浓度与颜色的散点图2")
            plt.show()
        elif i == "溴酸钾":
            group1 = data[i].iloc[:5, :]
            group2 = data[i].iloc[5:, :]
            group1 = group1.sort_values(by="ppm")
            group2 = group2.sort_values(by="ppm")
            x1, x2 = group1.iloc[:, 0], group2.iloc[:, 0]
            y1, y2 = group1.iloc[:, 1:], group1.iloc[:, 1:]
            for k in range(y1.shape[1]):
                plt.plot(x1, y1.iloc[:, k], "-o", markersize=4)
            scatter_cfg()
            plt.title(f"{i}的浓度与颜色的散点图1")
            plt.show()
            for j in range(y2.shape[1]):
                plt.plot(x2, y2.iloc[:, j], "-o", markersize=4)
            scatter_cfg()
            plt.title(f"{i}的浓度与颜色的散点图2")
            plt.show()
        elif i == "工业碱":
            group = data[i].sort_values(by="ppm")
            x = group.iloc[:, 0]
            y = group.iloc[:, 1:]
            for k in range(y.shape[1]):
                plt.plot(x, y.iloc[:, k], "-o", markersize=4)
            scatter_cfg()
            plt.title(f"{i}的浓度与颜色的散点图")
            plt.show()
        elif i=="奶中尿素":
            group1 = data[i].iloc[:5,:]
            group2 = data[i].iloc[5:10,:]
            group3 = data[i].iloc[10:15,:]
            x1,x2,x3 = group1.iloc[:,0],group2.iloc[:,0],group3.iloc[:,0]
            y1,y2,y3 = group1.iloc[:,1:],group2.iloc[:,1:],group3.iloc[:,1:]
            y = [y1,y2,y3]
            x = [x1,x2,x3]
            print(group1,group2,group3)
            cnt=0
            for yn in y:
                for k in range(yn.shape[1]):
                    plt.plot(x[cnt],yn.iloc[:,k],"-o",markersize =4)
                cnt+=1
                scatter_cfg()
                plt.title(f'{i}的浓度与颜色的散点图{cnt}')
                plt.show()


# 计算浓度与颜色RGB，HSV的相关系数
def rgb_hsv_related_coef(data1): ...


# 拟合浓度和R，G，B的结果
def fit_rgb(data1): ...


# 拟合浓度和H，S，V的结果
def fit_hsv(data1): ...
