from execute_data import get_data
from pprint import pprint


def main():
    data = get_data.read_csv1()
    # pprint(data)
    cv = get_data.coefficient_of_variation(data, "组胺")
    pprint(cv, width=150)
    # greys = get_data.greyscale_value(data)
    # pprint(greys)
    # cov = get_data.covariance(data, "组胺")
    # pprint(cov, width=300)
    deviation = get_data.deviation_hsv_rgb(data)
    pprint(deviation,width =150)


if __name__ == "__main__":
    main()
