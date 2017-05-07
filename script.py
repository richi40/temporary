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
for i in range(15):
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

  data = data.append(tmp)

for key in data.key.drop_duplicates():
  tmp = data[data.key == key]
  myplot((tmp['obs1'], tmp['obs2']), 'o')

# ----------------
# data processing
# ----------------
data = pd.read_csv(data_file)[['Y']]
data['OBS2'] = data.shift(-2) + 20
data.dropna(inplace=True)
data.reset_index(inplace=True)
data.rename(columns={'Y': 'obs1', 'index': 'T'}, inplace=True)

sns.set(font_scale=2)
#sns.set_style("ticks")
sns.despine(offset=10, trim=True)
plt.figure(figsize=(30, 10))
plt.plot(data['T'], data['obs1'], 'o', color='black')
plt.plot(data['T'], data['OBS2'], 'o', color='blue')
plt.show()

stan_data = {
  'N_t': len(data['T']),
  'N_pred': 3,
  'K': 3,
  'INIT_ALPHA': (data.ix[0, 'OBS2'] - data.ix[0, 'obs1']),
}
stan_data.update(data.to_dict('list'))

# ---------
# compile
# ---------
model = pystan.StanModel(file=model_file)

with open(model_obj_file, 'wb') as f:
  pickle.dump(model, f)

with open(model_obj_file, 'rb') as f:
  model = pickle.load(f)

# ----------
# sampling
# ----------
fit = model.sampling(data=stan_data, n_jobs=-1, iter=3000)

with open(fit_obj_file, 'wb') as f:
  pickle.dump(fit, f)

with open(fit_file, 'w') as f:
  f.write(str(fit))

with open(fit_obj_file, 'rb') as f:
  fit = pickle.load(f)

# --------------------------
# prepare for visualization
# --------------------------
sample_wide = fit.extract(permuted=False, inc_warmup=False)

sample_wide = pd.DataFrame(
  sample_wide.transpose(1,0,2).reshape(
    sample_wide.shape[0] * sample_wide.shape[1],
    sample_wide.shape[2]
  ), columns=[
    '{0}{1}[{2}]'.format(
      re.findall(r'^.*\[', c)[0].rstrip('['),
      re.findall(r'\[[0-9].*\]', c)[0].lstrip('[').rstrip(']').split(',')[1],
      re.findall(r'\[[0-9].*\]', c)[0].lstrip('[').rstrip(']').split(',')[0],
    )
    if c.find('[') != -1 and c.find(',') != -1 and c.find(']') != -1
    else c
    for c in fit.sim['fnames_oi']
  ]
)

sample_long = sample_wide.unstack().reset_index()
sample_long = sample_long.drop('level_1', axis=1)
sample_long.columns = ['param', 'sample']

chain_cnt = fit.sim['chains']
sample_cnt = int(len(sample_wide) / fit.sim['chains'])
param_cnt = len(fit.sim['fnames_oi'])
sample_long['chain'] = (
  np.ones((param_cnt, sample_cnt * chain_cnt), dtype='int64') *
  np.repeat(range(0, chain_cnt), sample_cnt)
).ravel()
sample_long['seq'] =sample_long.groupby(['param', 'chain']).cumcount()

# --------------
# visualization
# --------------
est_long = pd.DataFrame(
  np.percentile(sample_wide, q=[10, 50, 90], axis=0).T,
  columns=['low', 'middle', 'upper']
)
est_long['param'] = sample_wide.columns

est_long = est_long[
  (est_long['param'].str.find('[') != -1) & (est_long['param'].str.find(']') != -1)
].copy()

est_long['T'] = est_long.apply(
  lambda r: re.findall(r'\[[0-9]*\]', r['param'])[0].lstrip('[').rstrip(']'),
  axis=1
)

est_long['param'] = est_long.apply(
  lambda r: re.sub(r'\[[0-9]*\]', '', r['param']),
  axis=1
)

est_long['T'] = est_long['T'].astype('int64')
tmp = est_long['param'].str.find('pred_obs1') != -1
est_long.ix[tmp, 'T'] = est_long.ix[tmp, 'T'] + stan_data['N_t'] 

cols = est_long.columns.tolist()
cols.remove('param')
result = functools.reduce(
  lambda left, right: pd.merge(left, right, on='T', how='outer'), (
    data[['T', 'obs1']],
    data[['T', 'OBS2']],
    est_long.ix[est_long['param'] == 'trend', cols],
    est_long.ix[est_long['param'] == 'season', cols],
    est_long.ix[est_long['param'] == 'state', cols],
    est_long.ix[est_long['param'] == 'pred_obs1', cols],
  )
)

result.columns = [
  ['T', 'obs1', 'OBS2',] +
  [
    '{0}_{1}'.format(c, p)
    for c in ('trend', 'season', 'state', 'pred_obs1')
    for p in ('l', 'm', 'u')
  ]
]

sns.set(font_scale=2)
#sns.set_style("ticks")
sns.despine(offset=10, trim=True)
plt.figure(figsize=(30, 10))
plt.plot(result['T'], result['obs1'], 'o', color='black')
plt.plot(result['T'], result['OBS2'], 'o', color='blue')
plt.plot(result['T'], result['trend_m'], '-', color='black')
plt.plot(result['T'], result['state_m'], '--', color='black')
plt.errorbar(
  result['T'], result['pred_obs1_m'],
  yerr=[
    result['pred_obs1_m'] - result['pred_obs1_l'],
    result['pred_obs1_u'] - result['pred_obs1_m']
  ],
  color='black', ecolor='black', fmt='o'
)
plt.show()

# ---------------
# sampling check
# ---------------
sns.set(font_scale=2)
#sns.set_style("ticks")
sns.despine(offset=10, trim=True)
g = sns.FacetGrid(
  sample_long[
    sample_long['param'].isin(
      ['alpha[41]', 'beta0[41]', 'beta1[41]', 'beta2[41]', 's_obs2']
    )
  ],
  row="param",
#  col="param",
#  col_wrap=1,
  hue="chain",
  size=10, aspect=1,
  sharex=False, sharey=False
)
g = (g.map(sns.kdeplot, "sample").add_legend())
plt.show()

sns.set(font_scale=2)
sns.set_style("ticks")
sns.despine(offset=10, trim=True)
g = sns.FacetGrid(
  sample_long[
    sample_long['param'].isin(
      ['alpha[41]', 'beta0[41]', 'beta1[41]', 'beta2[41]', 's_obs2']
    )
  ],
  row="param",
  hue="chain",
  size=5, aspect=3,
  sharex=True, sharey=False
)
g = (g.map(plt.plot, "seq", "sample").add_legend())
plt.show()

sns.set(font_scale=2)
sns.set_style("ticks")
sns.despine(offset=10, trim=True)
g = sns.FacetGrid(
  sample_long,
  col="param",
  col_wrap=9,
  hue="chain",
  size=3, aspect=1,
  sharex=False, sharey=False
)
g = (g.map(sns.kdeplot, "sample").add_legend())
plt.show()

#sample_wide = sample_long[['seq', 'param', 'sample']].pivot_table(
#  index='seq', columns='param', values='sample'
#)
g = sns.PairGrid(sample_wide[[
  'alpha'
#  'alpha', 'beta[0]', 'beta[1]', 'beta[2]', 's_obs2'
]])
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_diag(sns.distplot, hist=False, fit=stats.norm);
plt.show()

sns.set(font_scale=2)
#sns.set_style("ticks")
sns.despine(offset=10, trim=True)
plt.figure(figsize=(30, 30))
sns.distplot(
  sample_wide[[
    'beta[0]'
  # 'alpha', 'beta[0]', 'beta[1]', 'beta[2]', 's_obs2'
  ]]
)
plt.show()

sample_long[
  sample_long['param'].isin(['alpha', 'beta[0]', 'beta[1]', 'beta[2]', 's_obs2'])
]




