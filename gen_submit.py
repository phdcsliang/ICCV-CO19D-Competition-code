import os

import pandas as pd

import numpy as np


res = pd.read_csv('results.csv')

covid = []
non = []

for i in range(len(res['FileName'])):
    if res['type'][i] == 2:
        covid.append(res['FileName'][i])
    else:
        non.append(res['FileName'][i])

with open('sub_covid.csv','w') as f:
    for i in range(len(covid)):
        f.write(str(covid[i]+','))
f.close()

with open('sub_non.csv','w') as f:
    for i in range(len(non)):
        f.write(str(non[i]+','))
f.close()
