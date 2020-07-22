import random
import numpy as np
import pandas as pd

print(random.uniform(-0.99,0.99))
data = np.zeros((10000,22))
columns = ['RSFMRI_C_NGD_CGC_NGD_CGC', 'RSFMRI_C_NGD_CGC_NGD_DT',
       'RSFMRI_C_NGD_CGC_NGD_DLA', 'RSFMRI_C_NGD_CGC_NGD_FO',
       'RSFMRI_C_NGD_CGC_NGD_SA', 'RSFMRI_C_NGD_CGC_NGD_VTA',
       'RSFMRI_C_NGD_DT_NGD_DT', 'RSFMRI_C_NGD_DT_NGD_DLA',
       'RSFMRI_C_NGD_DT_NGD_FO', 'RSFMRI_C_NGD_DT_NGD_SA',
       'RSFMRI_C_NGD_DT_NGD_VTA', 'RSFMRI_C_NGD_DLA_NGD_DLA',
       'RSFMRI_C_NGD_DLA_NGD_FO', 'RSFMRI_C_NGD_DLA_NGD_SA',
       'RSFMRI_C_NGD_DLA_NGD_VTA', 'RSFMRI_C_NGD_FO_NGD_FO',
       'RSFMRI_C_NGD_FO_NGD_SA', 'RSFMRI_C_NGD_FO_NGD_VTA',
       'RSFMRI_C_NGD_SA_NGD_SA', 'RSFMRI_C_NGD_SA_NGD_VTA',
       'RSFMRI_C_NGD_VTA_NGD_VTA', 'label']
	   
df = pd.DataFrame(columns = columns)

df['RSFMRI_C_NGD_CGC_NGD_CGC'] = np.random.normal(0.27, 0.1, size=(10000,))
df['RSFMRI_C_NGD_CGC_NGD_DT'] = np.random.normal(-0.09, 0.08, size=(10000,))
df['RSFMRI_C_NGD_CGC_NGD_DLA'] = np.random.normal(0.07, 0.07, size=(10000,))
df['RSFMRI_C_NGD_CGC_NGD_FO'] = np.random.normal(-0.01, 0.07, size=(10000,))
df['RSFMRI_C_NGD_CGC_NGD_SA'] = np.random.normal(0.12, 0.11, size=(10000,))
df['RSFMRI_C_NGD_CGC_NGD_VTA'] = np.random.normal(0.02, 0.07, size=(10000,))
df['RSFMRI_C_NGD_DT_NGD_DT'] = np.random.normal(0.24, 0.09, size=(10000,))
df['RSFMRI_C_NGD_DT_NGD_DLA'] = np.random.normal(-0.11, 0.08, size=(10000,))
df['RSFMRI_C_NGD_DT_NGD_FO'] = np.random.normal(0.06, 0.07, size=(10000,))
df['RSFMRI_C_NGD_DT_NGD_SA'] = np.random.normal(0.08, 0.1, size=(10000,))
df['RSFMRI_C_NGD_DT_NGD_VTA'] = np.random.normal(0.09, 0.07, size=(10000,))
df['RSFMRI_C_NGD_DLA_NGD_DLA'] = np.random.normal(0.24, 0.1, size=(10000,))
df['RSFMRI_C_NGD_DLA_NGD_FO'] = np.random.normal(0.05, 0.06, size=(10000,))
df['RSFMRI_C_NGD_DLA_NGD_SA'] = np.random.normal(-0.04, 0.1, size=(10000,))
df['RSFMRI_C_NGD_DLA_NGD_VTA'] = np.random.normal(-0.07, 0.07, size=(10000,))
df['RSFMRI_C_NGD_FO_NGD_FO'] = np.random.normal(0.2, 0.09, size=(10000,))
df['RSFMRI_C_NGD_FO_NGD_SA'] = np.random.normal(0.07, 0.1, size=(10000,))
df['RSFMRI_C_NGD_FO_NGD_VTA'] = np.random.normal(0.03, 0.06, size=(10000,))
df['RSFMRI_C_NGD_SA_NGD_SA'] = np.random.normal(0.38, 0.22, size=(10000,))
df['RSFMRI_C_NGD_SA_NGD_VTA'] = np.random.normal(0.08, 0.1, size=(10000,))
df['RSFMRI_C_NGD_VTA_NGD_VTA'] = np.random.normal(0.21, 0.1, size=(10000,))
labels = list('0'*7800 + '1'*2200)
random.shuffle(labels)
#df['label'] = np.random.randint(0,4,10000)
df['label'] = labels
df.to_csv("random_data.csv",index=False)

# for i in range(10000):
	# sampl = np.random.normal(0.2, 0.09, size=(21,))
	# sampl = np.append(sampl,random.randint(0,3))
	# data[i] = sampl

#pd.DataFrame(data, columns = columns).to_csv("random_data.csv")