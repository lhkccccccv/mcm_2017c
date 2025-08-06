from ..prelude import * 
__all__ = ["read_csv1", "coefficient_of_variation", "covariance", "greyscale_value"]


# 更改：rgb2hsv中，值的类型为pd.DataFrame.
# 读取data1中的数据，返回字典，键为物质的名称，值的类型为list，分别为RGB和HS的np.float
def read_csv1() -> dict:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Adjust based on project structure
    csv_path = project_root / "data" / "Data1.CSV"
    csv1 = pd.read_csv(csv_path, sep=",", encoding="utf-8")
    csv1.iloc[:, 1:6] = csv1.iloc[:, 1:6].apply(pd.to_numeric, errors="coerce")
    csv1 = csv1.dropna(subset=["ppm", "B", "G", "R", "H", "S"], how="all")
    cols = ["ppm", "B", "G", "R", "H", "S"]
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
        data[k] = pd.DataFrame(data[k], columns=cols)
        cols = ["ppm", "R", "G", "B", "H", "S"]
        data[k] = data[k][cols]
    return data


def standardization(material):
    """
    针对某一物质，取出RGBHS，进行标准化
    """
    tmp = material.iloc[:, 1:].to_numpy()
    means = np.mean(tmp, axis=0)
    standard = np.std(tmp, axis=0, ddof=1)
    tmp = (tmp - means) / standard
    return tmp


def corrcoef_matrix(data):
    matrix = {}
    for i in data:
        matrix[i] = data[i].corr()
        matrix[i] = matrix[i]['ppm']
        matrix[i].to_csv(
            f"./mcm_2017c/output/{i}的相关系数.csv", index=False, encoding="utf-8-sig" 
        )


def coefficient_of_variation(data) ->dict:
    """计算变异系数"""
    results = {}
    for name, df in data.items():
        grouped = df.groupby('ppm')
        cvs = grouped.agg(lambda x: x.std(ddof=0) / x.mean())
        cvs = cvs[grouped.size() > 1]
        if not cvs.empty:
            results[name] = cvs
    merge_df = (
    pd.concat(results, names=['物质', 'ppm'])
      .reset_index()
    )
    merge_df = merge_df.round(3)
    merge_df.to_csv(
        f"./mcm_2017c/output/变异系数.csv", index=False, encoding="utf-8-sig" 
    )
    # pprint(results)
    return results



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


def greyscale_value(data: dict) -> dict[str, pd.DataFrame]:
    """计算灰度值(rgb)  0.299R + 0.587G + 0.114B"""
    grey_values = {}
    for k, v in data.items():
        index = []
        values = []
        for row_idx in range(len(v)):
            row = data[k].iloc[row_idx, :]
            value = row["B"] * 0.114 + row["G"] * 0.587 + row["R"] * 0.299
            ppm = row["ppm"]
            index.append(ppm)
            values.append(value)
        grey_values[k] = pd.DataFrame(values, index=index, columns=["灰度值"])
    return grey_values


def deviation_hsv_rgb(data) -> dict[str, pd.DataFrame]:
    """
    计算HSV和RGB转换的误差(使用极坐标衡量),
    rgb转换结果 - 数据原给出结果 => 计算结果说明不符合  HSV的计算
    """
    deviation = {}
    for k0, v0 in data.items():
        gap = []
        index = []
        for row_idx in range(len(v0)):
            row = data[k0].iloc[row_idx, :]
            b, r, g = row["B"] / 255, row["R"] / 255, row["G"] / 255
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h, s, v = h * 255, s * 255, v * 255
            x0, y0 = row["S"] * math.cos(math.pi / 180 * row["H"]), row["S"] * math.sin(
                math.pi / 180 * row["H"]
            )
            x1, y1 = s * math.cos(math.pi / 180 * h), s * math.sin(math.pi / 180 * h)
            d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            gap.append([h - row["H"], s - row["S"], d])
            index.append(row["ppm"])
        deviation[k0] = pd.DataFrame(
            gap, index=index, columns=["H的误差", "S的误差", "色相和色调的计算误差值"]
        )
    return deviation


def pca(data, remain_component) -> dict:
    res = {}
    loads = []
    for i in data:
        tmp = standardization(data[i])
        pca = PCA(remain_component)
        res[i] = pca.fit_transform(tmp)
        # 获取方差解释率
        # pprint(f'{i}的主成分分析的方差解释率为')
        # pprint(pca.explained_variance_ratio_)
        # 输出pca载荷的信息
        df0 = pd.DataFrame(pca.components_,columns=["R","G","B","H","S"])
        df0['物质'] = i
        loads.append(df0)
    loads =pd.concat(loads,ignore_index=True)
    loads = loads[['物质',"R","G","B","H","S"]] 
    loads.to_csv('./mcm_2017c/output/pca_loads.csv', index=False, encoding='utf-8-sig')
    # 输出pca结果
    output = []
    for k,v in res.items():
        df = pd.DataFrame(v,columns=['特征1','特征2','特征3'])
        df['物质'] = k
        output.append(df)
    output = pd.concat(output, ignore_index=True)
    output = output[['物质','特征1','特征2','特征3']]
    output.to_csv('./mcm_2017c/output/pca_results.csv', index=False, encoding='utf-8-sig')
    return res


