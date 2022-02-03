#%%

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from matplotlib.pyplot import cm
import string

#%%

def plot_data(df, stat, color):
    xs = df[stat]
    bins = np.linspace(-3.0, 3.0, num=30)
    plt.hist(xs, bins=bins, color=color)
    plt.title(stat)
    # try:
    #     plt.savefig(f'{stat}_single_std_data.png')
    # except:
    #     pass
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
        x += 1
    return stat_dfs


def standerdize_data_length(stat_dfs):
    standerdizers = []
    x = 0
    for df in stat_dfs:
        standerdizers.append(len(stat_dfs[0]) / len(normalized_player_summed))
        stat_dfs[x] = df.sample(frac=standerdizers[x], replace=True)
        x += 1
    return stat_dfs


def plot_stat_data_by_std(stat_dfs, title):
    fig, axs = plt.subplots(4, 3)
    x = 0
    y = 0
    plt.title(title)
    bins = np.linspace(-3.0, 3.0, num=30)
    color = iter(cm.rainbow(np.linspace(0, 1, 16)))
    for df in stat_dfs:
        axs[x, y].hist(x=df, bins=bins, color=next(color))
        x += 1
        if x == 4:
            x = 0
            y += 1


#%%

csv = r"PATH_TO_DATASET"
df = pd.read_csv(csv)
df

#%%


for x in range(len(df)):
    #MP
    newmp = 0
    mp = df.loc[x, 'MP']
    mp = str(mp)
    mp = list(mp)
    mp.pop(-1)
    mp.pop(-1)
    if len(mp) == 1 and mp != ['n']:
        newmp = int(mp[0])
    elif len(mp) == 2:
        newmp = 10*int(mp[0]) + int(mp[1])
    elif len(mp) == 3:
        newmp = 60*int(mp[0]) + 10*int(mp[1]) + int(mp[2])
    elif len(mp) == 4:
        newmp = 600*int(mp[0]) + 60*int(mp[1]) + 10*int(mp[2]) + int(mp[3])
    else:
        newmp = 'NaN'
    df.at[x, 'MP'] = newmp
    #Age
    newage = 0
    age = df.loc[x, 'Age']
    age = str(age)
    age = list(age)
    age.pop(-1)
    age.pop(-1)
    newage = 3650*int(age[0]) + 365*int(age[1]) + 100*int(age[2]) + 10*int(age[3]) + int(age[4])
    df.at[x, 'Age'] = newage


#%%

df_means = df.mean()
print(df_means['MP'])
df['adj_ORtg'] = 0.0
for x in range(len(df)):
    adj_ortg = 0
    ortg = df.loc[x, 'ORtg']
    mp = df.loc[x, 'MP']
    adj_ortg = (ortg*mp) / (df_means['MP'])
    df.at[x, 'adj_ORtg'] = adj_ortg


#%%

df

#%%

currentplayer = df.loc[0, 'Name']
players = [currentplayer]
row = 1
while True:
    if df.loc[row, 'Name'] != currentplayer:
        currentplayer = df.loc[row, 'Name']
        players.append(currentplayer)
    row += 1
    if row == len(df) - 1:
        break


#%%

currentteam = df.loc[0, 'Tm']
teams = [currentteam]
row = 1
while True:
    if df.loc[row, 'Tm'] != currentteam:
        currentteam = df.loc[row, 'Tm']
        teams.append(currentteam)
    row += 1
    if row == len(df) - 1:
        break

#%%

teams

#%%

target_columns = ['Age', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', '+/-', 'ORtg', 'DRtg', 'BPM', 'adj_ORtg', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%']




#%%

player_summed = pd.DataFrame()
for stat in target_columns:
    player_summed[stat] = 0
player_summed['Name'] = ''
player_summed['Tm'] = ''
player_summed['PlayerID'] = 0
player_summed['Games'] = 0


#%%

for index, row in df.iterrows():
    print(row)
    break

#%%

df

#%%



#%%

PlayerID = 1001
z = 0
row = 0
currentplayer = df.loc[0, 'Name']
while True:
    player_rows = []
    while True:
        if df.loc[row, 'Name'] != currentplayer:
            player = df.loc[row - 1, 'Name']
            team = df.loc[row - 1, 'Tm']
            currentplayer = df.loc[row, 'Name']
            break
        player_rows.append(row)
        row += 1
        if row >= len(df):
            break
    if row >= len(df):
        break
    player_df = df[player_rows[0]:player_rows[-1]]
    player_df = player_df[target_columns]
    summed_stats = player_df.sum()
    #print(summed_stats)
    # if len(player_df) > 0:
    for stat in target_columns:
        player_summed.at[z, stat] = summed_stats[stat]
    player_summed.at[z, 'Name'] = player
    player_summed.at[z, 'Tm'] = team
    player_summed.at[z, 'PlayerID'] = PlayerID
    player_summed.at[z, 'Games'] = len(player_rows)
    PlayerID += 1
    z += 1



#%%



#%%

player_summed

#%%

team_summed = pd.DataFrame()
for team in teams:
    team_df = player_summed[player_summed['Tm'] == team]
    team_df = team_df.sum()
    team_summed[team] = team_df

#%%

team_summed

#%%

league_summed = player_summed.sum()

#%%

league_summed

#%%

target_columns

#%%

player_summed['FG%'] = player_summed['FG'] / player_summed['FGA']
player_summed['3P%'] = player_summed['3P'] / player_summed['3PA']
player_summed['FT%'] = player_summed['FT'] / player_summed['FTA']
player_summed['TS%'] = player_summed['PTS'] / (player_summed['FGA'] + (0.44 * player_summed['FTA']))
player_summed['eFG%'] = (player_summed['FG'] + (0.5 * player_summed['3P'])) / player_summed['FGA']



#%%

pg_columns = []
for column in target_columns:
    statname = column + '_pg'
    pg_columns.append(statname)
    player_summed[statname] = player_summed[column] / player_summed['Games']


#%% md



#%%



#%%

player_summed_means = player_summed.mean()
player_summed_stdevs = player_summed.std()
normalized_player_summed = player_summed.copy()
for column in target_columns:
    normalized_player_summed[column] = (normalized_player_summed[column] - player_summed_means[column]) / player_summed_stdevs[column]
for column in pg_columns:
    normalized_player_summed[column] = (normalized_player_summed[column] - player_summed_means[column]) / player_summed_stdevs[column]




color = iter(cm.rainbow(np.linspace(0, 1, len(target_columns) + len(pg_columns))))
for stat in target_columns:
    plot_data(normalized_player_summed, stat, next(color))

for stat in pg_columns:
    plot_data(normalized_player_summed, stat, next(color))


#%%

dfs = data_by_std(normalized_player_summed, 'MP')
dfs
big_stat_array = []
x=0
for stat in target_columns:
    big_stat_array.append(gather_target_stat_data(stat, normalized_player_summed))
    big_stat_array[x] = standerdize_data_length(big_stat_array[x])
    plot_stat_data_by_std(big_stat_array[x], stat)
    # try:
    #     plt.savefig(f'{stat}_single_std_data.png')
    # except:
    #     pass
    filenum = str(x)
    x+=1

#%%

big_stat_array = []
x=0
for stat in pg_columns:
    big_stat_array.append(gather_target_stat_data(stat, normalized_player_summed))
    big_stat_array[x] = standerdize_data_length(big_stat_array[x])
    plot_stat_data_by_std(big_stat_array[x], stat)
    # try:
    #     plt.savefig(f'{stat}_single_std_data.png')
    # except:
    #     pass
    filenum = str(x)
    x+=1

#%%

# factor = (2/3) - (0.5 * (league_summed['AST'] / league_summed['FG'])) / (2 * (league_summed['FG'] / league_summed['FT']))
# VOP = league_summed['PTS'] / (league_summed['FG'] - league_summed['ORB'] + league_summed['TOV'] + 0.44 * league_summed['FTA'])
# DRB = (league_summed['TRB'] - league_summed['ORB']) / league_summed['TRB']

#%%

# def calc_uPER(factor, VOP, DRB, MP, three_pt, AST, team_AST, team_FG, TOV, FGA, FG, FTA, FT, PF, TRB, ORB, STL, BLK, lg_FT, lg_PF, lg_FTA):
#     if MP == 0:
#         return 0
#     uPER = ((1.0 / MP) *
#            (three_pt
#           + (2/3) * AST
#           + (2 - factor * (team_AST / team_FG)) * FG
#           + (FT *0.5 * (1 + (1 - (team_AST / team_FG)) + (2/3) * (team_AST / team_FG)))
#           - VOP * TOV
#           - VOP * DRB * (FGA - FG)
#           - VOP * 0.44 * (0.44 + (0.56 * DRB)) * (FTA - FT)
#           + VOP * (1 - DRB) * (TRB - ORB)
#           + VOP * DRB * ORB
#           + VOP * STL
#           + VOP * DRB * BLK
#           - PF * ((lg_FT / lg_PF) - 0.44 * (lg_FTA / lg_PF) * VOP)))
#     print(uPER)
#     return uPER

#%%

# per_player_stats = ['MP', '3P', 'AST', 'TOV', 'FGA', 'FG', 'FTA', 'FT', 'PF', 'TRB', 'ORB', 'STL', 'BLK']

#%%

# lg_FT = league_summed['FT']
# lg_PF = league_summed['PF']
# lg_FTA = league_summed['FTA']
# player_summed['uPER'] = 0.0

#%%

# per_stat_dict = {}
# for i in range(len(player_summed)):
#     try:
#         for stat in per_player_stats:
#             per_stat_dict[stat] = player_summed.loc[i, stat]
#         team = player_summed.loc[i, 'Tm']
#         team_AST = team_summed.loc['AST', team]
#         team_FG = team_summed.loc['FG', team]
#         uPER = calc_uPER(factor, VOP, DRB, per_stat_dict['MP'], per_stat_dict['3P'], per_stat_dict['AST'], team_AST, team_FG,
#                          per_stat_dict['TOV'], per_stat_dict['FGA'], per_stat_dict['FG'], per_stat_dict['FTA'], per_stat_dict['FT'],
#                          per_stat_dict['PF'], per_stat_dict['TRB'], per_stat_dict['ORB'], per_stat_dict['STL'], per_stat_dict['BLK'],
#                          lg_FT, lg_PF, lg_FTA)
#         player_summed.at[i, 'uPER'] = uPER
#     except:
#         player_summed.at[i, 'uPER'] = 0



#%%



#%%

# player_summed[player_summed['uPER'] > 0.008]

#%%

# league_summed

