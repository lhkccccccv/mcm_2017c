from ..prelude import *
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

# 拟合浓度和R，G，B，H，S的结果
def fit_rgb(data1): 
    ...

def fit_graph(x,y,material):
    def residuals(params, X_pca, y):
        w = params[:-1]       # 回归权值向量 w，shape = (k,)
        b = params[-1]        # 截距 b
        y_pred = X_pca.dot(w) + b  # 预测值，shape = (n_samples,)
        return y_pred - y         # 残差，shape = (n_samples,)
    initial_guess = np.zeros(4)
    result, _ = leastsq(residuals, initial_guess, args=(x, y))
    w_fit = result[:-1]  # 最优的回归权值向量 w
    b_fit = result[-1]   # 最优的截距 b
    idx = np.argsort(x[:, 0])
    for i in range(3):
        xp = x[:, i][idx] # 第i+1主成分
        yp = (w_fit[0] * x[:,0] + 
            w_fit[1] * x[:, 1].mean() +w_fit[2]*x[:,2]+ 
            b_fit)
        # plt.scatter(x[:, 0], y, label="真实点",s=4)
        # plt.plot(xp, yp, 'r-', label=f"{material}线性拟合第{i+1}个主成分结果")
        # plt.gca().xaxis.set_major_formatter(
        #         FormatStrFormatter("%.1f")
        #     )  # x轴保留1位小数
        # plt.gca().yaxis.set_major_formatter(
        #     FormatStrFormatter("%.2f")
        # )  # y轴保留2位小数
        # plt.xlabel(f"主成分{i+1}")
        # plt.ylabel("浓度")
        # plt.legend()
        # plt.show()
    return w_fit,b_fit

def retrograde_coef(data,w_fit,b_fit,material):
    tmp = get_data.standardization(data[material])
    X_mean = np.mean(data[material].iloc[:,1:],axis=0)
    X_std = np.std(data[material].iloc[:,1:], axis=0, ddof=1)
    pca = PCA(3)
    pca.fit_transform(tmp)
    beta = pca.components_.T @ w_fit/X_std
    beta0 = b_fit - (X_mean / X_std) @ (pca.components_.T @ w_fit)
    print(f"{material}的w_fit为{w_fit}")
    print(f"b_fit为{b_fit}")
    y =  data[material].iloc[:,1:]@ beta + beta0
    return y
