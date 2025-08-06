from . import *

def main():
    data = get_data.read_csv1()
    # pprint(data)

    # 相关系数的计算结果
    # pprint(get_data.corrcoef_matrix(data))

    # 主成分分析的结果
    res =get_data.pca(data,3)
    #pprint(res)
    retrograde =[]
    idx = []
    for i in res:
        idx.append(f"{i}的逆推浓度")
        w_fit,b_fit = draw.fit_graph(res[i],data[i]['ppm'],i)   
        ppm = draw.retrograde_coef(data,w_fit,b_fit,i)
        retrograde.append(ppm)
    df = pd.DataFrame(retrograde,index=idx)
    df.to_csv("./mcm_2017c/output/逆推浓度.csv",index=True,header=False,encoding='utf-8-sig')
    # 画出浓度与颜色的散点图
    # draw.scatter_coef(data)

    # 变异系数：
    # cv = get_data.coefficient_of_variation(data)

    # 解释hsv与rgb转换不成立的函数
    # deviation = get_data.deviation_hsv_rgb(data)


if __name__ == "__main__":
    main()
