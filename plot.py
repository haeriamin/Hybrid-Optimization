# Copyright (C) 2020 Amin Haeri [ahaeri92@gmail.com]

import csv
import matplotlib.pyplot as plt


def plot(opt_time):
    path = './output/'
    filename = 'mape' + str(opt_time) + '.csv'
    f = open(path + filename, 'r')
    reader = csv.reader(f)
    firstline = True
    c = 0
    mape = []
    for row in reader:
        c += 1
        if c != 1:
            mape.append(float(row[0]))
    f.close()

    plt.figure()
    plt.plot(mape)# 'bo', markersize=1, 'r--', linewidth=2
    plt.xlabel("Function evaluation #")
    plt.ylabel("MAPE %")
    # plt.ylim(0, 400)
    plt.savefig(path + 'mape' + str(opt_time) + '.png', dpi=1000)
    plt.show()
