#!/usr/bin/env python
# -*- coding utf-8 -*-

# -------
# import
# -------
import os
import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# pd.options.display.max_rows = 999

# ------------
# environment
# ------------
current_date = datetime.date.today() - relativedelta(months=1)

file_dir = os.getcwd()
# file_dir =os.path.abspath(os.path.dirname(__file__))
input_dir = os.path.join(file_dir, 'input')

if not os.path.exists(input_dir):
  os.mkdir(input_dir) 

raw_data_file = os.path.join(input_dir, 'raw_data.csv')
processed_data_file = os.path.join(input_dir, 'processed_data.csv')

# ----------------
# common function
# ----------------
def diff_month(d1, d2):
  return (d1.year - d2.year)*12 + d1.month - d2.month + 1

# ----------------
# data processing
# ----------------
data = pd.read_csv(raw_data_file)
data = data[data['key'] != 'key']
keys = data.key.drop_duplicates()
sampled_keys = np.random.choice(keys, 100, replace=False)
data = data[data.key.isin(sampled_keys)]
data = data[['key', 'init_date', 'init_year', 'init_month', 'yyyymm', 'obs1', 'obs2']]
data.rename(columns={'init_date': 'init_yyyymmdd'}, inplace=True)

data['init_date'] = data['init_yyyymmdd'].apply(
  lambda v: datetime.datetime.strptime(str(v), '%Y-%m-%d').date()
)

data['yyyymm_date'] = data['yyyymm'].apply(
  lambda v: datetime.datetime.strptime(str(v), '%Y%m').date()
)

data['relative_months'] = data.apply(
  lambda r: diff_month(r['yyyymm_date'], r['init_date']), axis=1
)

data.to_csv(processed_data_file, index=False)

