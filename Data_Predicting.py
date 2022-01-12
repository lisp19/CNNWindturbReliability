import numpy as np
import os
import time
import pandas as pd
import pickle

T1 = time.time()
data = pd.read_csv('Source/train_labels.csv')
print(data.info())
print(data.head(20))
datasets = {}
data = data[data['f_id'] == 12]
n = len(data)
for i in range(n):
	path ='Source/' + '0' + str(data.iloc[n - i - 1,0]) + '/' + data.iloc[n - i - 1, 1]
	if os.path.exists(path):
		if data.iloc[n - i - 1,2] != 1.0 and  data.iloc[n - i - 1,2] != 0.0:
			datasets[data.iloc[n - i - 1, 1]] = pd.read_csv(path)
			pass
		pass
	pass

print(len(datasets))
T2 = time.time()
print('The dataloading needs %s seconds.' % (T2-T1))
with open('Data/datasets_predicting.pkl','wb') as file:
	pickle.dump(datasets,file)
print('Saved to local disk, consuming %s seconds' % (time.time() - T2))
