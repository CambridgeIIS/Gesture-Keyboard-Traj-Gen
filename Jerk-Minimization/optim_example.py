#!/usr/bin/env python
# coding=utf-8

import numpy as np
from traj_gen import optim_trajectory as opt_t
import time
import pandas as pd

    
if __name__ == '__main__':



    keys = []
    x_pos = []
    y_pos = []
    f = open('holokeyboard.txt', 'r')
    str = f.readline()
    str = f.readline()
    while len(str) > 1:
        info = str[:-1].split(';')
        keys.append(info[0])
        x_pos.append(int(info[1]))
        y_pos.append(int(info[2]))

        str = f.readline()

    df = pd.DataFrame({
        'keys': keys,
        'x_pos': x_pos,
        'y_pos': y_pos,
    })

    df.to_csv('holokeyboard.csv')
    df = pd.read_csv("holokeyboard.csv", index_col='keys')
    

    phrase = 'that'

    dim = 2
    knots = np.array([0.0, 7.0])
    pntDensity = 5
    objWeights = np.array([0, 0,0,1])
    optTraj = opt_t.OptimTrajGen(knots, dim, pntDensity)

    ts = np.linspace(0, 7.0, num=len(phrase))


    Xs = []
    for char in phrase:
        Xs.append([df.loc[char][1],df.loc[char][2]])


    Xs = np.stack(Xs)
    print(Xs)
    Xdot = np.array([0, 0])
    Xddot = np.array([0, 0])

    # create pin dictionary
    for i in range(Xs.shape[0]):
        pin_ = {'t':ts[i], 'd':0, 'X':Xs[i]}
        optTraj.addPin(pin_)
    pin_ = {'t':ts[0], 'd':1, 'X':Xdot,}
    optTraj.addPin(pin_)
    pin_ = {'t':ts[-1], 'd':2, 'X':Xddot,}
    optTraj.addPin(pin_)

    # solve
    optTraj.setDerivativeObj(objWeights)
    print("solving")
    time_start = time.time()
    optTraj.solve()
    time_end = time.time()
    print(time_end - time_start)

    # # plot
    # ## showing trajectory
    print("trajectory")
    optTraj.showTraj(4)
    print('path')
    fig_title = 'minimzed derivatives order: {}'.format(np.where(optTraj.weight_mask>0)[0].tolist())
    Xs = optTraj.showPath(fig_title)