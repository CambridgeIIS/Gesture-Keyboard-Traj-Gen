#!/usr/bin/env python
# coding=utf-8

import numpy as np
from traj_gen import poly_trajectory as pt
import time

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
    
    dim = 2
    knots = np.array([0.0, 7.0])
    pntDensity = 5
    objWeights = np.array([0, 0,0,1])

    ts = np.linspace(0, 7.0, num=len(phrase))


    Xs = []
    for char in phrase:
        Xs.append([df.loc[char][1],df.loc[char][2]])


    Xs = np.stack(Xs)
    print(Xs)
    Xdot = np.array([0, 0])
    Xddot = np.array([0, 0])

    order = 8
    optimTarget = 'end-derivative' #'end-derivative' 'poly-coeff'
    maxConti = 4
    pTraj = pt.PolyTrajGen(knots, order, optimTarget, dim, maxConti)



    # create pin dictionary
    for i in range(Xs.shape[0]):
        pin_ = {'t':ts[i], 'd':0, 'X':Xs[i]}
        pTraj.addPin(pin_)
    pin_ = {'t':ts[0], 'd':1, 'X':Xdot,}
    pTraj.addPin(pin_)
    pin_ = {'t':ts[0], 'd':2, 'X':Xddot,}
    pTraj.addPin(pin_)

    # solve
    pTraj.setDerivativeObj(objWeights)
    print("solving")
    time_start = time.time()
    pTraj.solve()
    time_end = time.time()
    print(time_end - time_start)

    # plot
    ## showing trajectory
    print("trajectory")
    pTraj.showTraj(4)
    print('path')
    fig_title ='poly order : {0} / max continuity: {1} / minimized derivatives order: {2}'.format(pTraj.N, pTraj.maxContiOrder, np.where(pTraj.weight_mask>0)[0].tolist())
    pTraj.showPath(fig_title)