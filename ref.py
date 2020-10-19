# Copyright (C) 2020 Amin Haeri [ahaeri92@gmail.com]

import csv


def get_exp(depth):
    time = 20;22  # Exp time [sec] 45/60
    fr_exp = 62.5  # Sampling frequency [Hz]
    start_exp = 1
    path = './input/'

    filename = 'weight_calibration.csv'
    f = open(path + filename, 'r')
    reader = csv.reader(f)
    firstline = True
    c = 0
    for row in reader:
        c += 1
        if c != 1:
            Fx0 = float(row[0])
            Fy0 = float(row[1])
            Fz0 = float(row[2])
            if c == 2:
                f.close()
                break

    if depth == 0.02:
        filename = 'run2_moving_sand_load_2cm_40mms_changed.csv'
    elif depth == 0.05:
        filename = 'run7_moving_load_ramp_50_800_50_changed.csv'
    else:
        print('Depth is not valid')
        return 0

    f = open(path + filename, 'r')
    reader = csv.reader(f)
    firstline = True
    c = 0
    F_exp = {'Fx': [], 'Fy': [], 'Fz': []}
    for row in reader:
        c += 1
        if c != 1:
            F_exp['Fx'].append(float(row[0]) - Fx0)
            F_exp['Fy'].append(float(row[1]) - Fy0)
            F_exp['Fz'].append(float(row[2]) - Fz0)
    f.close()

    return F_exp
