## -*- coding: utf-8 -*-
#"""
#Spyder Editor
#

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import scipy.optimize as solver
import datetime as dt
from functools import reduce

Closeprice = pd.DataFrame()
tickers = ['AAPL','TSM','COKE','V','GE','JNJ','T','BABA']

l = len(tickers)
records = 0
for i in tickers:
    tmp = web.DataReader(i, 'yahoo', '1/1/2010', dt.date.today())
    Closeprice[i] = tmp['Adj Close']
    if records == 0:
        records = len(tmp)
    else:
        records = min(records, len(tmp))



returns = np.log(Closeprice / Closeprice.shift(1))
print(returns.head())
returns = returns.tail(records - 1)
print(returns.head())
returns.fillna(value=0, inplace=True)
mean = returns.mean() * 252
cov = returns.cov() * 252
print(mean)
sds = []
rtn = []

for _ in range(200000):
    w = np.random.rand(l)
    w /= sum(w)
    rtn.append(sum(mean * w))
    sds.append(np.sqrt(reduce(np.dot, [w, cov, w.T])))

plt.plot(sds, rtn, 'bo') 

##function we need to minimize
def sd(w):
    return np.sqrt(reduce(np.dot, [w, cov, w.T]))

x0 = np.array([1.0 / l for x in range(l)])
##weight range from 0-1
bounds = tuple((0, 1) for x in range(l))

##range of returns
given_r = np.arange(-.07, .20, .005)
##list for minimum volatility
risk = []

for i in given_r:
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * mean) - i}]
    outcome = solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds)
    risk.append(outcome.fun)
plt.plot(risk, given_r, 'r-x')


constraints = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
minv = solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds).fun
minvr = sum(solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds).x * mean)
plt.plot(minv, minvr, 'w^')
plt.title('Efficient Frontier:')
plt.xlabel('portfolio volatility')
plt.ylabel('portfolio return')



