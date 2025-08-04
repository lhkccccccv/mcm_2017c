import pandas as pd
import numpy as np
import colorsys
import math
from pathlib import Path
from pprint import pprint
from sklearn.decomposition import PCA

__all__ = ["read_csv1","coefficient_of_variation","covariance","greyscale_value"]

# 更改：rgb2hsv中，值的类型为pd.DataFrame.
# 读取data1中的数据，返回字典，键为物质的名称，值的类型为list，分别为RGB和HS的np.float
def read_csv1() -> dict:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Adjust based on project structure
    csv_path = project_root / "data" / "Data1.CSV"
    csv1 = pd.read_csv(csv_path, sep=",", encoding="utf-8")
    csv1.iloc[:, 1:6] = csv1.iloc[:, 1:6].apply(pd.to_numeric, errors="coerce")
    csv1 = csv1.dropna(subset=["ppm", "B", "G", "R", "H","S"], how="all")
    cols = ['ppm','B','G','R','H','S']
    data = {}
    # data_cols = csv1.columns[1:6]
    for i, x in enumerate(csv1.iloc[:, 0]):
        if pd.notna(x):
            data[x] = []
    tmp = ""
    for r, i in enumerate(csv1.iloc[:, 0]):
        if pd.notna(i):
            tmp = i
        else:
            i = tmp
        data[i].append(csv1.iloc[r, 1:7].tolist())
    tmp = 0
    for i in range(len(data["硫酸铝钾"])):
        current = data["硫酸铝钾"][i]
        if pd.notna(current[0]):
            tmp = current[0]
        else:
            current[0] = tmp
    for k in data:
        data[k] = pd.DataFrame(data[k],columns=cols)
        cols = ['ppm','R','G','B','H','S']
        data[k] = data[k][cols]
    return data


def coefficient_of_variation(data, material) -> np.ndarray:
    """计算变异系数"""
    assert material in data, f"{material} 应该是data中的物质,请重新输入"
    df = data[material]
    means = np.mean(df, axis=0)
    assert (means != 0).all(), f"{means} 这里应该不含有0(as denom)"

    # 总体标准差计算
    # standard = np.std(df,ddof=0,axis=0)

    # 样本标准差计算
    standard = np.std(df, ddof=1, axis=0)
    cv = standard / means
    return cv


def covariance(data, material) -> np.ndarray:
    """计算样本协方差"""
    assert material in data, f"{material} 应该是data中的物质,请重新输入"
    df = data[material]
    df = np.array(df)
    means = np.mean(df, axis=0)
    col_size = df.shape[1]
    cov = np.zeros((col_size, col_size), dtype=float)
    for i in range(col_size):
        for k in range(i, col_size):
            res = np.sum((df[:, i] - means[i]) * (df[:, k] - means[k]))
            cov[i][k] = res / (float(len(df)) - 1)  # 样本协方差
            if i != k:
                cov[k][i] = cov[i][k]
    return cov


def greyscale_value(data:dict)->dict[str,pd.DataFrame]:
    """计算灰度值(rgb)  0.299R + 0.587G + 0.114B"""
    grey_values = {}
    for k,v in data.items():
        index = []
        values = []
        for row_idx in range(len(v)):
            row = data[k].iloc[row_idx,:]
            value = row['B']*0.114+row['G']*0.587+row['R']*0.299
            ppm = row['ppm']
            index.append(ppm)
            values.append(value)
        grey_values[k] = pd.DataFrame(values,index=index,columns=["灰度值"])
    return grey_values


def deviation_hsv_rgb(data)->dict[str,pd.DataFrame]:
    """
    计算HSV和RGB转换的误差(使用极坐标衡量),
    rgb转换结果 - 数据原给出结果 => 计算结果说明不符合  HSV的计算
    """
    deviation ={}
    for k0,v0 in data.items():
        gap=[]
        index =[]
        for row_idx in range(len(v0)):
            row = data[k0].iloc[row_idx,:]
            b,r,g =row['B']/255,row['R']/255,row['G']/255
            h,s,v = colorsys.rgb_to_hsv(r,g,b)
            h,s,v = h*255,s*255,v*255
            x0,y0 = row['S']*math.cos(math.pi/180*row['H']),row['S']*math.sin(math.pi/180*row['H'])
            x1,y1 = s*math.cos(math.pi/180*h),s*math.sin(math.pi/180*h)
            d=math.sqrt((x1-x0)**2+(y1-y0)**2)
            gap.append([h-row['H'],s-row['S'],d])
            index.append(row['ppm'])
        deviation[k0] = pd.DataFrame(gap,index =index,columns=['H的误差','S的误差','色相和色调的计算误差值'])
    return deviation

def pca(data,remain_component)->dict:
    res = {}
    for i in data:
        # 取出RGBHS，进行标准化
        tmp = data[i].iloc[:,1:].to_numpy()
        means = np.mean(tmp,axis=0)
        standard = np.std(tmp,axis=0,ddof=1)
        tmp = (tmp-means)/standard
        pca = PCA(remain_component)
        res[i] = pca.fit_transform(tmp)
        print(f"{i}的载荷如下")
        pprint(pca.components_)
    return res

# 计算浓度与颜色RGB，HSV的相关系数
def rgb_hsv_related_coef(data):
    ...

# 拟合浓度和R，G，B的结果
def fit_rgb(data):
    ...

# 拟合浓度和H，S，V的结果
def fit_hsv(data):
    ...


