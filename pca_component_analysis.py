import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
df = pd.read_csv('winequality-red.csv', sep=';')
#df = df.sample(n=100, random_state=42)
scaler = StandardScaler()
scaled = scaler.fit_transform(df.drop('quality', axis=1))

PCA = PCA(n_components=5)
pca_result = PCA.fit_transform(scaled)

pc_df = pd.DataFrame(data=pca_result,
                     columns=['PC1', 'PC2', 'PC3', 'PC4'])
pc_df = pd.concat([pc_df, df['quality'].reset_index(drop=True)], axis=1)
names = '"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'.split(';')

g = sns.pairplot(pc_df, hue='quality', diag_kind='kde')