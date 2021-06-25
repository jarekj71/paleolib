#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:51:50 2021

@author: jarekj
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def rint(min,max,size=1):
    return np.random.randint(min,max,size)

def rfloat(min,max,size=1):
    return (max - min) * np.random.random(size) + min

class random_param_search():

    def __init__(self,learner,nsearch=20):

        self._X = None
        self._y = None
        self._nsearch = nsearch

        self._learner = learner
        self._params = {}
        self._best = None

    def set_data(self,X,y,columns=None):
        self._X = X
        self._y = y
        if columns is None:
            try:
                self._columns = X.columns
            except:
                self._columns = None

    def set_nsearch(self,nsearch):
        self._nsearch = nsearch

    def tuned_parameter(self,param,rmin,rmax,dtype='float'):
        if dtype == 'int':
            fun = rint
        else:
            fun = rfloat
        self._params[param]=fun(rmin,rmax,self._nsearch)
    
    def fixed_parameters(self,**kwargs):
        for key, value in kwargs.items():
            self._params[key] = value

    def remove_param(self,param):
        try:
            self._params.pop(param)
        except:
            print("No such param {}".format(param))

    def search_for_best_parameters(self,scoring,n_iter=100,cv=5,n_jobs=1):
        if self._X is None:
            return

        clf = RandomizedSearchCV(self._learner,
                           self._params,cv=cv,
                           scoring=scoring,
                           n_iter = n_iter,
                           n_jobs=n_jobs)
        clf.fit(self._X,self._y)
        print(clf.best_params_)
        self._best = clf.best_params_

    def get_best_parameters(self):
        return self._best