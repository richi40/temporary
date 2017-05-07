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

processed_data_csv = os.path.join(input_dir, 'processed_data.csv')
trend_season_stan = os.path.join(model_dir, 'trend_season.stan')
trend_season_pkl = os.path.join(model_dir, 'trend_season.pkl')
trend_season_fit_pkl = os.path.join(output_dir, 'trend_season_fit.pkl')
trend_season_fit_txt = os.path.join(output_dir, 'trend_season_fit.txt')

# ----------------
# common function
# ----------------
def myplot(var=None, marker='-'):
  sns.set(font_scale=2)
  plt.figure(figsize=(30, 15))
  
  for o in var:
    plt.plot(o, marker)

  plt.show()

# -------------
# data to stan
# -------------
data = pd.read_csv(processed_data_csv)

stan_data = {
  'N_obs': len(data['obs1'].dropna()),
  'N_state': 3,
  'I_state': 3,
  'OBS': 3,
}
stan_data.update(data.to_dict('list'))







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



# ==================================================
# シミュレーションデータの作成
# 収縮期血圧をシミュレーションします
# ==================================================

# ------------------------------
# 薬を投与しなかった時の「状態」
# ------------------------------
N_NotMedicine = 10              # 薬を投与しなかった日数
N_Medicine = 100                # 薬を投与した日数
N = N_NotMedicine + N_Medicine  # 合計日数
muZero = 160                    # 初期の血圧の「状態」

# 血圧の「状態」のシミュレーション
mu = np.empty(N, dtype=float)
mu.fill(np.nan)
mu[0] = stats.norm.rvs(
  loc=muZero, scale=3, size=1
)

for i in range(1, N):
  mu[i] = stats.norm.rvs(
    loc=mu[i-1], scale=3, size=1
  )

# --------------------------------------------------
# 時間によって変化する（慣れる）薬の効果のシミュレーション
# --------------------------------------------------
# 薬を使っても、徐々に血圧は下がらなくなっていく
coefMedicineTrendZero = 0.005

# 時間的に変化する薬の効果
# 薬の効果をトレンドモデルで表す
coefMedicineTrend = np.empty(N_Medicine, dtype=float)
coefMedicineTrend.fill(np.nan)
coefMedicineTrend[0] = stats.norm.rvs(
  loc=coefMedicineTrendZero, scale=0.03, size=1
)

# トレンドのシミュレーション
for i in range(1, N_Medicine):
  coefMedicineTrend[i] = stats.norm.rvs(
    loc=coefMedicineTrend[i-1], scale=0.03, size=1
  )

plt.plot(coefMedicineTrend); plt.show() 


# 薬の効果のシミュレーション
# 薬の効果の初期値
coefMedicineZero = -25
coefMedicine = np.empty(N_Medicine, dtype=float)
coefMedicine.fill(np.nan)
coefMedicine[0] = stats.norm.rvs(
  loc=coefMedicineTrend[0] + coefMedicineZero, scale=0.5, size=1
)

for i in range(1, N_Medicine):
  coefMedicine[i] = stats.norm.rvs(
    loc=coefMedicineTrend[i] + coefMedicine[i-1], scale=0.5, size=1
  )

plt.plot(coefMedicine); plt.show() 


# 実際の薬の効果は、さらにノイズが加わるとする
coefMedicineReal = np.empty(N_Medicine, dtype=float)
coefMedicineReal.fill(np.nan)

for i in range(0, N_Medicine):
  coefMedicineReal[i] = stats.norm.rvs(
    loc=coefMedicine[i], scale=2, size=1
  )

# -------------------------------
# 血圧の観測値のシミュレーション
# -------------------------------
# 最初の10日は薬なし
# 70日後に薬を倍にした
# 100日後に薬を3倍にした
medicine = np.r_[
  np.repeat(0, N_NotMedicine),
  np.repeat(1, 60), np.repeat(2, 30), np.repeat(3, 10) 
]

bloodPressure = np.empty(N, dtype=float)
bloodPressure.fill(np.nan)

bloodPressureMean = np.empty(N, dtype=float)
bloodPressureMean.fill(np.nan)

# 最初の10日は薬なし
for i in range(0, N_NotMedicine):
  bloodPressureMean[i] = mu[i]
  bloodPressure[i] = stats.norm.rvs(
    loc=bloodPressureMean[i], scale=10, size=1
  )

# 薬を投与した後の血圧のシミュレーション
for i in range(N_NotMedicine, N):
  bloodPressureMean[i] = mu[i] + coefMedicineReal[i-N_NotMedicine] * medicine[i]
  bloodPressure[i] = stats.norm.rvs(
    loc=bloodPressureMean[i], scale=10, size=1
  )

for obj in (
  mu,
  coefMedicineTrend,
  coefMedicine,
  coefMedicineReal,
  bloodPressureMean,
  bloodPressure,
):
  plt.plot(obj)

for i in (9, 69, 99):
  plt.axvline(x=i,color='green')

plt.show()

for obj in (
  mu,
  bloodPressureMean,
  bloodPressure,
):
  plt.plot(obj)

for i in (9, 69, 99):
  plt.axvline(x=i,color='green')

plt.show()

