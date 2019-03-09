# main libraries
import pandas as pd
import numpy as np
import time

# visual libraries
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

# sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
from sklearn.externals import joblib
from sklearn.decomposition import PCA

def display_ghg(x, pos):
    return '{:03.0f}'.format(x)


#Fetch the data
df = pd.read_csv('AnnualEnergyConsumption2016.csv')

print(df.head())
print(df.columns)

# Top 10 Address Graphs

# GHG vs. Address
formatter = FuncFormatter(display_ghg)
ghg_df = df.sort_values(by=['GHG Emissions(Kg)'], ascending=[0])

fig, ax = plt.subplots(figsize=(15, 10))
ax.yaxis.set_major_formatter(formatter)
ax.set_ylabel("GHG Emissions(Kg)")
ax.set_xlabel("Address")

ghg_df = ghg_df[:7]
print(ghg_df.head())

barlist=plt.bar(ghg_df['Address'], ghg_df['GHG Emissions(Kg)'])
plt.savefig('Plots/Top10Graphs/ghg.png')
plt.clf()


# Water Usage vs. Address
formatter = FuncFormatter(display_ghg)
ghg_df = df.sort_values(by=['Annual Flow (Mega Litres)'], ascending=[0])
ghg_df = ghg_df[:7]

fig, ax = plt.subplots(figsize=(15, 10))
ax.yaxis.set_major_formatter(formatter)
ax.set_ylabel("Water Usage (Mega Litres)")
ax.set_xlabel("Address")
print(ghg_df.head())

barlist=plt.bar(ghg_df['Address'], ghg_df['Annual Flow (Mega Litres)'])
plt.savefig('Plots/Top10Graphs/water.png')
plt.clf()

# Electricity vs. Address
formatter = FuncFormatter(display_ghg)
ghg_df = df.sort_values(by=['Electricity Quantity'], ascending=[0])
ghg_df = ghg_df[:7]

fig, ax = plt.subplots(figsize=(15, 10))
ax.yaxis.set_major_formatter(formatter)
ax.set_ylabel("Electricity(kWh)")
ax.set_xlabel("Address")
print(ghg_df.head())

'''
col_list = ['r','g', 'b', 'c', 'm', 'y', 'k']
ind = 0
row = 0
comment_color = dict()
#print(comment_color.keys())
#print(ghg_df['Comments'][row])
#print(ghg_df['Comments'][row] in comment_color.keys())
colors = []
for i in range(7):
    print('index: ', i)
    print(ghg_df.head())
    print('Comment: ', ghg_df['Comments'][i])
    test = ghg_df['Comments'][i] in comment_color.keys()
    print(colors)
    print(test)
    if test:
        print('in if', test)
        key = ghg_df['Comments'][i]
        cur_color = comment_color[key]
    else:
        cur_color = col_list[ind]
        comment_color[ghg_df['Comments'][i]] = cur_color
        ind = ind + 1

    colors.append(cur_color)
'''

barlist = plt.bar(ghg_df['Address'], ghg_df['Electricity Quantity'], color=colors)

for i in range(len(barlist)):

    barlist[i].set_color()
plt.savefig('Plots/Top10Graphs/electricity.png')
plt.clf()
