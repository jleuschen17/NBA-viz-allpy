

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from matplotlib.pyplot import cm



csv = r"C:\Users\jleus\OneDrive\Documents\CSVS\league2011cleaned.csv"
df = pd.read_csv(csv)
df



def plot_data(df, stat, color):
    xs = df[stat]
    bins = np.linspace(-3.0, 3.0, num=30)
    plt.hist(xs, bins=bins, color=color)
    plt.title(stat)
    plt.show()



def data_by_std(df, std_stat):
    stds = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    dfs = []
    for std in stds:
        dfs.append(df[df['MP'] > std])
    return dfs





def gather_target_stat_data(target_stat, df):
    stat_dfs = []
    x = 0
    for df in dfs:
        stat_dfs.append(df[target_stat])
        stat_dfs[x].dropna()
        x+=1
    return stat_dfs




def standerdize_data_length(stat_dfs):
    standerdizers = []
    x = 0
    for df in stat_dfs:
        standerdizers.append(len(stat_dfs[0]) / len(df))
        stat_dfs[x] = df.sample(frac=standerdizers[x], replace=True)
        x+=1
    return stat_dfs







def plot_stat_data_by_std(stat_dfs, title):
    fig, axs = plt.subplots(4, 3)
    x=0
    y=0
    plt.title(title)
    bins = np.linspace(-3.0, 3.0, num=30)
    color = iter(cm.rainbow(np.linspace(0, 1, 16)))
    for df in stat_dfs:
        axs[x, y].hist(x=df, bins=bins, color=next(color))
        x+=1
        if x == 4:
            x=0
            y+=1



csv = r"C:\Users\jleus\OneDrive\Documents\CSVS\league2011cleaned.csv"
df = pd.read_csv(csv)







stats = []
for stat in df:
    stats.append(stat)
print(stats)



stats = ['Age', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', '+/-', 'TS%', 'eFG%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM']



color = iter(cm.rainbow(np.linspace(0, 1, len(stats))))
for stat in stats:
    plot_data(df, stat, next(color))



dfs = data_by_std(df, 'MP')
big_stat_array = []
x=0
for stat in stats:
    big_stat_array.append(gather_target_stat_data(stat, df))
    big_stat_array[x] = standerdize_data_length(big_stat_array[x])
    plot_stat_data_by_std(big_stat_array[x], stat)
    filenum = str(x)
    x+=1




