from .execute_data import get_data,draw
from pprint import pprint


def main():
    data = get_data.read_csv1()
    # pprint(data)

    # 相关系数的计算结果
    # pprint(get_data.corrcoef_matrix(data))
    
    # pprint(cv, width=150)

    # 主成分分析的结果
    res =get_data.pca(data,3)


    # 画出浓度与颜色的散点图
    # draw.scatter_coef(data)


    # greys = get_data.greyscale_value(data)
    # pprint(greys)
    # cov = get_data.covariance(data, "组胺")
    # pprint(cov, width=300)
    # deviation = get_data.deviation_hsv_rgb(data)
    # pprint(deviation,width =150)


if __name__ == "__main__":
    main()
