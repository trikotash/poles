#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 23:35:46 2022

@author: trikotash
"""
import numpy as np
from scipy.optimize import minimize_scalar

glob_tech1 = np.array([1,0],dtype = 'double')
glob_tech2 = np.array([0,345],dtype = 'double')


def f(x,y):
    return (x-1)*(x-1) + y*y

def one_dim_tech_to_min(l):
    return f(glob_tech1[0]+l*glob_tech2[0],glob_tech1[1]+l*glob_tech2[1])

def minimize(chi,bound, n):
    u1 = np.array([1,0],dtype = 'double')
    u2 = np.array([0,-1],dtype = 'double')
    init =  np.array([-500,1000],dtype = 'double')
    l_tech = 0
    x = chi
    tech1 = np.zeros(2)
    tech2 = np.zeros(2)
    tech3 = np.zeros(2)
    for i in range(n):
        global glob_tech1
        global glob_tech2
        glob_tech1 = np.copy(init)
        glob_tech2 = np.copy(u1)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (0,bound),method = 'bounded').x       
        tech1 =np.copy(init + l_tech*u1)
        glob_tech1 = np.copy(tech1)
        glob_tech2 = np.copy(u2)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (0,bound),method = 'bounded').x
        tech2 = np.copy(tech1 + l_tech*u2)
        u1 = np.copy(u2)
        u2 = np.copy(tech2 - init)
        glob_tech1 = np.copy(init)
        glob_tech2 = np.copy(u2)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (0,bound),method = 'bounded').x
        init = init + l_tech*u2
    return init

print(1)
print(minimize(1000,2000,100))



