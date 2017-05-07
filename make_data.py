#!/usr/bin/env python
# -*- coding utf-8 -*-

# -------
# import
# -------
import os
import re
import math
import pickle
import pystan
import datetime
import functools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from dateutil.relativedelta import relativedelta

# pd.options.display.max_rows = 999

# ------------
# environment
# ------------
current_date = datetime.date.today() - relativedelta(months=1)

file_dir = os.getcwd()
# file_dir =os.path.abspath(os.path.dirname(__file__))
input_dir = os.path.join(file_dir, 'input')
model_dir = os.path.join(file_dir, 'model')
output_dir = os.path.join(file_dir, 'output')

for dir in (input_dir, model_dir, output_dir):
  if not os.path.exists(dir):
    os.mkdir(dir) 

data_file = os.path.join(input_dir, 'data.csv')
model_file = os.path.join(model_dir, 'stan.model')
model_obj_file = os.path.join(model_dir, 'model.pkl')
fit_obj_file = os.path.join(output_dir, 'fit.pkl')
fit_file = os.path.join(output_dir, 'fit.txt')

# ----------------
# common function
# ----------------
def diff_month(d1, d2):
  return (d1.year - d2.year)*12 + d1.month - d2.month + 1

def myplot(var=None, marker='-'):
  sns.set(font_scale=2)
  plt.figure(figsize=(30, 15))
  
  for o in var:
    plt.plot(o, marker)

  plt.show()

# ----------
# make data
# ----------
def make_data(
  key=1,
  init_obs1=400,
  init_date=datetime.datetime.strptime('201402', '%Y%m').date(),
  slope=-0.1,
  month_ptn = 0,
  obs1_std_changes = [10],
  eff_lag=[
    [0.25, 0.25, 0.25, 0.25],
    [0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4],
  ],
  margin= 0.2,
  na_rate = 0.2,
):

  data = pd.DataFrame({
    'key' : key,
    'init_obs1' : init_obs1,
    'init_date' : init_date,
    'init_year' : init_date.year,
    'init_month' : init_date.month,
    'yyyymm' : [
      (init_date + relativedelta(months=cnt)).strftime('%Y%m')
      for cnt in range(0, diff_month(current_date, init_date))
    ]
  }, columns=[
    'key',
    'init_obs1',
    'init_date',
    'init_year',
    'init_month',
    'yyyymm'
  ])

  data['attenuation_rate'] = data.reset_index()['index'].apply(
    lambda t: np.exp(slope * t)
  )
  data.ix[data['attenuation_rate'] <= 0.1, 'attenuation_rate'] = 0.1 

  data['trend'] = data['attenuation_rate'] * data['init_obs1']

  data['month'] = data['yyyymm'].str[-2:].astype(int)
  data['flag'] = 1

  tmp = pd.pivot_table(
    data, index='yyyymm', columns='month', values='flag'
  ).fillna(0)

  for c in [c for c in range(1, 13) if c not in tmp.columns]:
    tmp[c] = 0.0
  tmp = tmp[[c for c in range(1, 13)]]
  
  month_rate = np.sin(np.linspace(-np.pi, np.pi, 13))[0:12] * 0.3
  month_rate = np.r_[month_rate[month_ptn:], month_rate[0:month_ptn]]
  month_rate = np.outer(data['trend'].values, month_rate)
  data['eff_month'] = np.sum(tmp.values * month_rate, axis=1)

  data['state'] = data['trend'] + data['eff_month']

  tmp = data.loc[:, ['state']]
  for k in range(1, np.array(eff_lag).shape[0]+1):
    tmp['sft{0}'.format(k)] = tmp['state'].shift(k)

  tmp.dropna(inplace=True)

  index = tmp.index

  tmp = np.array_split(
    tmp.values,
    np.array(eff_lag).shape[0] 
  )

  ar = np.dot(tmp[0], eff_lag[0])
  for i in range(1, len(tmp)): 
    ar = np.append(ar, np.dot(tmp[i], eff_lag[i]))

  tmp = pd.DataFrame(ar, columns=['ar'])
  tmp.index = index
  data = pd.merge(data, tmp, right_index=True, left_index=True, how='left')

  tmp = np.repeat(obs1_std_changes, len(data)/len(obs1_std_changes))
  stds = np.r_[tmp, np.repeat(obs1_std_changes[-1], len(data) - len(tmp))]
  data['obs1'] = stats.norm.rvs(
    loc=data['state'], scale=stds, size=len(data)
  )

  data['obs2'] = stats.norm.rvs(
    loc=data['state'] + data['ar'] + data['state'] * margin,
    scale=stds.min(),
    size=len(data)
  )

  index = np.random.choice(data.index, int(len(data)*na_rate), replace=False)
  data.ix[index, 'obs1'] = np.nan
  index = np.random.choice(data.index, int(len(data)*na_rate), replace=False)
  data.ix[index, 'obs2'] = np.nan

  return data


init_obs1_samples = stats.norm.rvs(loc=500, scale=200, size=10**5) 
init_obs1_samples = init_obs1_samples[
  (100 < init_obs1_samples) & (init_obs1_samples < 5000)
]

start_date = datetime.datetime.strptime('201001', '%Y%m').date()
init_date_samples = [
  (start_date + relativedelta(months=cnt)).strftime('%Y%m')
  for cnt in range(0, diff_month(current_date, start_date))
]

slope_samples = stats.norm.rvs(loc=-0.025, scale=0.1, size=10**5) 
slope_samples = slope_samples[(-0.05 < slope_samples) & (slope_samples < -0.01)]

data = pd.DataFrame()
for i in range(6*10**4):
  if i % 1000 == 0:
    print('--- {0} ---'.format(i))
  tmp = make_data(
    key=i,
    init_obs1=int(np.random.choice(init_obs1_samples)),
    init_date=datetime.datetime.strptime(
      np.random.choice(init_date_samples), '%Y%m'
    ).date(),
    slope=np.random.choice(slope_samples),
    month_ptn = np.random.choice(range(0, 12)),
    obs1_std_changes = [10],
    eff_lag=[
      [0.25, 0.25, 0.25, 0.25],
      [0.4, 0.3, 0.2, 0.1],
      [0.1, 0.2, 0.3, 0.4],
    ],
    margin= 0.2,
    na_rate = 0.2,
  )

  tmp.to_csv('raw_data.csv', index=False, mode='a')


for key in data.key.drop_duplicates():
  tmp = data[data.key == key]
  myplot((tmp['obs1'], tmp['obs2']), 'o')


