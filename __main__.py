from .execute_data import get_data
from pprint import pprint


def main():
    data = get_data.read_csv1()
    # pprint(data)
    cv = get_data.coefficient_of_variation(data, "组胺")
    print("组胺的变异系数")
    pprint(cv, width=150)
    res =get_data.pca(data,2)
    # greys = get_data.greyscale_value(data)
    # pprint(greys)
    # cov = get_data.covariance(data, "组胺")
    # pprint(cov, width=300)
    # deviation = get_data.deviation_hsv_rgb(data)
    # pprint(deviation,width =150)


if __name__ == "__main__":
    main()
