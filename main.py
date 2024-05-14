# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:31:17 2023

@author: thoma
"""

import pandas as pd
from shape import Shape,finder_multi,finder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from scipy.stats import ttest_rel,ttest_1samp
from pmdarima.arima import auto_arima
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
import seaborn as sns
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from scipy import stats
from keras import backend as K
import os
import random
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


### Replication - set seed 

seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

################################################ Analysis 

df_migra = pd.read_csv('Data\Migra_new.csv',index_col=0,parse_dates=True)
df_fp = pd.read_csv('Data\Food.csv',index_col=0,parse_dates=True)
df_conf = pd.read_csv('Data\Conf.csv',index_col=0,parse_dates=True)

df_fp = df_fp.loc[:df_migra.index[-1],:]
df_conf = df_conf.loc[:df_migra.index[-1],:]


data = df_migra.iloc[:-9,:]
cov=[df_fp,df_conf]

pred_tot=[]
for col in range(76):
    try:
        shape = Shape()
        shape.set_shape(df_migra.iloc[-9:-3,col]) 
        shape_cov1 = Shape()
        shape_cov1.set_shape(df_fp.iloc[-9:-3,col]) 
        shape_cov2 = Shape()
        shape_cov2.set_shape(df_conf.iloc[-9:-3,col]) 
        shape_cov = [shape_cov1,shape_cov2]
        
        find = finder_multi(data,cov,shape,shape_cov)
        find.find_patterns(min_d=3,select=True,metric='dtw')
        pred_ori = find.predict(horizon=3,plot=False,mode='mean')
        df_fill = pd.DataFrame([shape.values[-1],shape.values[-1],shape.values[-1]])
        df_fill= df_fill.T
        df_fill.columns = pred_ori.columns
        pred = pd.concat([df_fill,pred_ori],axis=0)
        pred.index = df_migra.iloc[-4:,col].index
        
        pred_ori = pred_ori*(df_migra.iloc[-9:-3,col].max()-df_migra.iloc[-9:-3,col].min())+df_migra.iloc[-9:-3,col].min()
        pred_tot.append(pred_ori)
    except:
        pass

data = df_migra.iloc[:-9,:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_1 = MinMaxScaler(feature_range=(0, 1))
scaler_2 = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

df_fp_d = scaler_1.fit_transform(df_fp)
df_conf_d = scaler_1.fit_transform(df_conf)
df_fp = df_fp_d[-len(data_scaled)-9:-9,:]
df_conf = df_conf_d[-len(data_scaled)-9:-9,:]

n_steps = 6
X_lstm, y_lstm = [], []
for i in range(len(data_scaled) - n_steps - 2):
    X_data = data_scaled[i:i + n_steps, :]
    X_fp = df_fp[i:i + n_steps, :]
    X_conf = df_conf[i:i + n_steps, :]
    X_combined=np.stack([X_data.T, X_fp.T, X_conf.T], axis=1)
    X_lstm.append(X_combined)
    y_lstm.append(data_scaled[i + n_steps:i + n_steps + 3, :])


X_lstm, y_lstm = np.concatenate(X_lstm), np.array(y_lstm)
X_lstm = X_lstm.transpose((0, 2, 1))
y_lstm = y_lstm.reshape((y_lstm.shape[0]*y_lstm.shape[2], 3))
model = Sequential()
model.add(LSTM(units=10, activation='relu', input_shape=(n_steps, 3),return_sequences=True))
model.add(LSTM(units=10, activation='relu'))
model.add(Dense(units=3))  
model.compile(optimizer='adam', loss='mse')  

train_size = int(len(X_lstm) * 0.80)  # Adjust the split ratio as needed
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]
model.fit(X_train, y_train, epochs=100, batch_size=16)#,validation_data=(X_test, y_test))

pred_x = df_migra.iloc[-9:-3,:]
pred_x = scaler.transform(pred_x)
df_fp = df_fp_d[-9:-3,:]
df_conf = df_conf_d[-9:-3,:]

X_lstm = np.stack([pred_x, df_fp, df_conf], axis=1)
X_lstm = X_lstm.T
X_lstm = X_lstm.transpose((0, 2, 1))
predictions = model.predict(X_lstm)

# Inverse transform the predictions to get them in the original scale
predictions_original_scale = scaler.inverse_transform(predictions.T)
predictions_original_scale[predictions_original_scale<0]=0


df_migra = pd.read_csv('Data\Migra_new.csv',index_col=0,parse_dates=True)
df_fp = pd.read_csv('Data\Food.csv',index_col=0,parse_dates=True)
df_conf = pd.read_csv('Data\Conf.csv',index_col=0,parse_dates=True)

data = df_migra.iloc[:-3,:]
df_fp = df_fp.loc[:df_migra.index[-1],:]
df_conf = df_conf.loc[:df_migra.index[-1],:]


pred_ar=[]
for i in range(len(data.columns)):
    ts = pd.concat([data.iloc[:,i].reset_index(drop=True),df_fp.iloc[-len(data.iloc[:,i])-6:-6,i].reset_index(drop=True),df_conf.iloc[-len(data.iloc[:,i])-6:-6,i].reset_index(drop=True)],axis=1)
    exog = pd.concat([df_fp.iloc[-6:-3,i].reset_index(drop=True),df_conf.iloc[-6:-3,i].reset_index(drop=True)],axis=1)
    model_f = auto_arima(ts.iloc[:,0],X=ts.iloc[:,1:])
    pred_ar.append(model_f.predict(3,X=exog))
pred_ar=pd.DataFrame(pred_ar)
pred_ar=pred_ar.T




mse_lstm=[]
mse_sf=[]
mse_ar=[]
std=[]
for i in range(76):
    mse_lstm.append(mean_squared_error(predictions_original_scale[:,i],df_migra.iloc[-3:,i]))
    mse_sf.append(mean_squared_error(pred_tot[i].iloc[:,0],df_migra.iloc[-3:,i]))
    mse_ar.append(mean_squared_error(pred_ar.iloc[:,i],df_migra.iloc[-3:,i]))
    std.append(abs(pred.iloc[1:,0]-pred.iloc[1:,1]).mean())
    
ttest_rel(pd.Series(mse_sf),pd.Series(mse_ar))
series_1= pd.Series(mse_lstm)-pd.Series(mse_sf)
series_2= pd.Series(mse_ar)-pd.Series(mse_sf)


model=pd.DataFrame([mse_ar,mse_lstm,mse_sf]).T
model_name=['ARIMA','LSTM','ShapeFinder']
df_test= gpd.read_file('Data/Shape/SS_adm2.shp')

best=[]
obs_v=[]
pred_v=[]
for i in range(76):
    best.append(model_name[model.idxmin(axis=1)[i]])
    obs_v.append(df_migra.iloc[-3:,i].sum())
    pred_v.append(pred_tot[i].iloc[:,0].sum())

df_m = pd.DataFrame([best,obs_v,pred_v,df_migra.columns]).T

df_test = pd.merge(df_test,df_m,left_on='ADM2_EN',right_on=3, how='left')
df_test = df_test.rename(columns={0:'best',1:'obs',2:'pred'})




################################################ Figures 

##################
# Figure 6
##################

model_no_z=pd.DataFrame([np.log((model.iloc[:,1]+1)/(model.iloc[:,2]+1)),np.log((model.iloc[:,0]+1)/(model.iloc[:,2]+1))])
model_no_z=model_no_z.T
model_no_z.columns=["ratio_1", "ratio_2"]
mean_1 = model_no_z['ratio_1'].mean()
mean_2 = model_no_z['ratio_2'].mean()

print(mean_1)
print(mean_2)

melted_df = pd.melt(model_no_z, value_vars=["ratio_1", "ratio_2"], var_name="Model", value_name="Log ratio")


sns.set(style="ticks",rc={"figure.figsize": (7, 8)})
b = sns.boxplot(data = melted_df,           
                    x = "Model",       # x axis column from data
                    y = "Log ratio",       # y axis column from data
                    width = 0.4,        # The width of the boxes
                    color = "white",  # Box colour
                    linewidth = 2,      # Thickness of the box lines
                    showfliers = False)  # Sop showing the fliers
b = sns.stripplot(data = melted_df,           
                    x = "Model",       # x axis column from data
                    y = "Log ratio",     # y axis column from data
                      color = "darkgrey", # Colours the dots
                      linewidth = 1,     # Dot outline width
                      alpha = 0.4)       # Makes them transparent
b.set_ylabel("Log Ratio", fontsize = 20)
b.set_xlabel("Model", fontsize = 20)
b.set_xticklabels(['LSTM', 'ARIMA'])
b.tick_params(axis='both', which='both', labelsize=20)
b.text(x=0, y=7.5, s="P-val < 0.001", ha='center', va='center', fontsize=20, color='black')
b.text(x=1, y=7.5, s="P-val < 0.001", ha='center', va='center', fontsize=20, color='black')
b.axhline(y=0, linestyle='--', color='black', linewidth=1)
sns.despine(offset = 5, trim = True)
plt.show()

##################
# Figure 7
##################

# Define custom colormap colors
colors = [(0.512, 0.512, 0.512),(0.294, 0.294, 0.294),(0.737, 0.737, 0.737)]  # Blue, Green, Red
values = [0, 0.5, 1]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(values, colors)))

fig, axs = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [4, 1]})
ax_map = axs[0]
df_test.plot(column='best', ax=ax_map, cmap=custom_cmap, legend=False, legend_kwds={'bbox_to_anchor': (1, 1)}, missing_kwds={
        "color": "white",
        "edgecolor": "lightgrey",
        "hatch": "///",
        "label": "No migration",
    })
ax_map.set_xlabel('Longitude', fontsize=25)
ax_map.set_ylabel('Latitude', fontsize=25)
ax_map.tick_params(axis='both', which='both', labelsize=20)
ax_map.spines['top'].set_visible(False)
ax_map.spines['right'].set_visible(False)
ax_bar = axs[1]
category_counts = df_test['best'].value_counts()
category_counts['No migration'] = 3
sorted_categories = category_counts.sort_index().index
sorted_counts = category_counts.sort_index().values
colors = [(0.512, 0.512, 0.512),(0.294, 0.294, 0.294),(1,1,1),(0.737, 0.737, 0.737)]
bars = ax_bar.bar(sorted_categories, sorted_counts, color=colors)
bars[2].set_hatch('///')
bars[2].set_edgecolor('lightgrey')
for bar in bars:
    height = bar.get_height()
    ax_bar.annotate(f'N={height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=20)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_visible(False)
ax_bar.yaxis.set_visible(False)
ax_bar.tick_params(axis='x', rotation=45)
ax_bar.tick_params(axis='both', which='both', labelsize=20)
plt.tight_layout()
plt.show()


##################
# Figure 2
# col 42 correspond to Melut May-2022
# and the find.sequences[0] is Guit June 2021
##################
data = df_migra.iloc[:-8,:]
col=42
flag=False
while flag==False:
    if (df_migra.iloc[-8:-3,col]==0).all()==False:
        try:
            shape = Shape()
            shape.set_shape(df_migra.iloc[-8:-3,col]) 
            shape_cov1 = Shape()
            shape_cov1.set_shape(df_fp.iloc[-10:-3,col]) 
            shape_cov = [shape_cov1]
            find = finder_multi(data,cov,shape,shape_cov)
            find.find_patterns(min_d=0.5,select=True,metric='dtw')
            if find.sequences != []:
                flag=True
        except:
            pass
    col+=1
    
data_to_plot = df_migra.iloc[-8:-3, col - 1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_migra.iloc[-8:-3, col - 1], marker='o',markersize=10,linewidth=3)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=40)
years = data_to_plot.index.year
ax.set_xticks(data_to_plot.index[[0,2,4]])
ax.set_xticklabels(data_to_plot.index[[0,2,4]].strftime('%b-%y'))
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = find.sequences[0][0]
ax.plot(find.sequences[0][0],marker='o',markersize=10,linewidth=3)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=40)
ax.set_xticks(data_to_plot.index[[0,2,4]])
ax.set_xticklabels(data_to_plot.index[[0,2,4]].strftime('%b-%y'))
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = df_migra.iloc[-8:, col - 1]
ax.plot(df_migra.iloc[-8:,col-1],marker='o',markersize=10,linewidth=3)
ax.plot(df_migra.iloc[-3:,col-1],color='red',marker='o',markersize=10,linewidth=6)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=40)
ax.set_xticks(data_to_plot.index[[0,2,4,6]])
ax.set_xticklabels(data_to_plot.index[[0,2,4,6]].strftime('%b-%y'))
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = df_migra.loc[find.sequences[0][0].index[0]:,find.sequences[0][0].name].iloc[:8]
ax.plot(df_migra.loc[find.sequences[0][0].index[0]:,find.sequences[0][0].name].iloc[:8],marker='o',markersize=10,linewidth=3)
ax.plot(df_migra.loc[find.sequences[0][0].index[-1]:,find.sequences[0][0].name].iloc[1:4],color='red',marker='o',markersize=10,linewidth=6)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=40)
ax.set_xticks(data_to_plot.index[[0,2,4,6]])
ax.set_xticklabels(data_to_plot.index[[0,2,4,6]].strftime('%b-%y'))
plt.show()



fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = df_fp.iloc[-10:-3,col-1]
ax.plot(df_fp.iloc[-10:-3,col-1],color='green',marker='o',markersize=10,linewidth=3)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=40)
ax.set_xticks(data_to_plot.index[[0,2,4,6]])
ax.set_xticklabels(data_to_plot.index[[0,2,4,6]].strftime('%b-%y'))
plt.show()

data_to_plot = find.sequences_cov[0][0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(find.sequences_cov[0][0].iloc[1:],color='green',marker='o',markersize=10,linewidth=3)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=40)
ax.set_xticks(data_to_plot.index[[0,2,4,6]])
ax.set_xticklabels(data_to_plot.index[[0,2,4,6]].strftime('%b-%y'))
plt.show()

##################
# Figure Appendix C1 
##################

df_migra = pd.read_csv('Data/Migra_new.csv',index_col=0,parse_dates=True)
df_fp = pd.read_csv('Data/Food.csv',index_col=0,parse_dates=True)
df_conf = pd.read_csv('Data/Conf.csv',index_col=0,parse_dates=True)

df_fp = df_fp.loc[:df_migra.index[-1],:]
df_conf = df_conf.loc[:df_migra.index[-1],:]

data = df_migra.iloc[:-9,:]
cov=[df_fp,df_conf]

tot_seq=[]
tot_cov1=[]
tot_cov2=[]
shape_l=[]
shape_cov1_l=[]
shape_cov2_l=[]
dist_l=[]
for col in range(76):
    if (df_migra.iloc[-9:-3,col]==0).all()==False:
        shape = Shape()
        shape.set_shape(df_migra.iloc[-9:-3,col]) 
        shape_cov1 = Shape()
        shape_cov1.set_shape(df_fp.iloc[-9:-3,col]) 
        shape_cov2 = Shape()
        shape_cov2.set_shape(df_conf.iloc[-9:-3,col]) 
        shape_cov = [shape_cov1,shape_cov2]
        find = finder_multi(data,cov,shape,shape_cov)
        find.find_patterns(min_d=3,select=True,metric='dtw') 
        sequences = find.sequences
        sequences_cov = find.sequences_cov
        sorted_indices = sorted(range(len(sequences)), key=lambda k: sequences[k][1])
        sorted_sequences = [sequences[i] for i in sorted_indices]
        sorted_sequences_cov_1 = [sequences_cov[0][i] for i in sorted_indices]
        sorted_sequences_cov_2 = [sequences_cov[1][i] for i in sorted_indices]
        tot_seq.append([k for k,j in sorted_sequences[:10]])
        dist_l.append([j for k,j in sorted_sequences[:10]])
        tot_cov1.append([k for k in sorted_sequences_cov_1[:10]])
        tot_cov2.append([k for k in sorted_sequences_cov_2[:10]])
        shape_l.append(df_migra.iloc[-8:-3,col])
        shape_cov1_l.append(df_fp.iloc[-9:-3,col])
        shape_cov2_l.append(df_conf.iloc[-9:-3,col])


sorted_indices = sorted(range(len(dist_l)), key=lambda i: dist_l[i][0])[:6]
result_tot_seq = [tot_seq[i] for i in sorted_indices]
result_dist_l = [dist_l[i] for i in sorted_indices]
result_shape_l = [shape_l[i] for i in sorted_indices]
result_shape_cov1_l = [shape_cov1_l[i] for i in sorted_indices]
result_shape_cov2_l = [shape_cov2_l[i] for i in sorted_indices]
result_tot_cov1 = [tot_cov1[i] for i in sorted_indices]
result_tot_cov2 = [tot_cov2[i] for i in sorted_indices]

fig = plt.figure(figsize=(30, 25))
num_rows = 15 + 6  # 18 original rows + 6 spacer rows
num_cols = 13
width_ratios = [2, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
gs = GridSpec(num_rows, num_cols, figure=fig, width_ratios=width_ratios)

# Function to configure ax properties
def configure_ax(ax):
    ax.set(xticks=[], yticks=[])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Plotting loop
for j in range(0, 15, 3):
    print(j)
    actual_row = j + j // 3  # Adjust row for spacers

    # Row for shape_l
    ax_text = fig.add_subplot(gs[actual_row, 0])
    #ax_text.text(0, 0.25, f"Shape {j//3}", rotation=0, fontsize=25)  # Placeholder text
    ax_text.set(xticks=[], yticks=[])
    configure_ax(ax_text)

    ax_shape = fig.add_subplot(gs[actual_row, 1])
    ax_shape.plot(result_shape_l[j//3], color='red',linewidth=3)
    if j==0:
        ax_shape.spines['top'].set_visible(False)
        ax_shape.spines['right'].set_visible(False)
        ax_shape.spines['bottom'].set_visible(False)
        ax_shape.spines['left'].set_visible(False)
        ax_shape.set(xticks=[], yticks=[])
    else:
        configure_ax(ax_shape)

    for i in range(10):
        ax_seq = fig.add_subplot(gs[actual_row, i + 3])
        alpha = -result_dist_l[j//3][i]+min(result_dist_l[j//3])+1 if result_dist_l[j//3][i] <= 2 * min(result_dist_l[j//3]) else 0
        ax_seq.plot(result_tot_seq[j//3][i], color='black', alpha=alpha,linewidth=3)
        configure_ax(ax_seq)

    # Row for shape_cov1_l
    actual_row_cov1 = actual_row + 1
    ax_text_cov1 = fig.add_subplot(gs[actual_row_cov1, 0])
    ax_text_cov1.text(0, 0.25, result_shape_cov1_l[j//3].name, rotation=0, fontsize=60, weight='bold')  # Placeholder text
    configure_ax(ax_text_cov1)

    ax_shape_cov1 = fig.add_subplot(gs[actual_row_cov1, 1])
    ax_shape_cov1.plot(result_shape_cov1_l[j//3], color='green',linewidth=3)
    configure_ax(ax_shape_cov1)

    for i in range(10):
        ax_cov1 = fig.add_subplot(gs[actual_row_cov1, i + 3])
        alpha = -result_dist_l[j//3][i]+min(result_dist_l[j//3])+1 if result_dist_l[j//3][i] <= 2 * min(result_dist_l[j//3]) else 0
        ax_cov1.plot(result_tot_cov1[j//3][i], color='black', alpha=alpha,linewidth=3)
        configure_ax(ax_cov1)

    # Row for shape_cov2_l
    actual_row_cov2 = actual_row_cov1 + 1
    ax_text_cov2 = fig.add_subplot(gs[actual_row_cov2, 0])
    #ax_text_cov2.text(0, 0.25, f"Cov2 {j//3}", rotation=0, fontsize=25)  # Placeholder text
    configure_ax(ax_text_cov2)

    ax_shape_cov2 = fig.add_subplot(gs[actual_row_cov2, 1])
    ax_shape_cov2.plot(result_shape_cov2_l[j//3], color='orange',linewidth=3)
    configure_ax(ax_shape_cov2)

    for i in range(10):
        ax_cov2 = fig.add_subplot(gs[actual_row_cov2, i + 3])
        alpha = -result_dist_l[j//3][i]+min(result_dist_l[j//3])+1 if result_dist_l[j//3][i] <= 2 * min(result_dist_l[j//3]) else 0
        ax_cov2.plot(result_tot_cov2[j//3][i], color='black', alpha=alpha,linewidth=3)
        configure_ax(ax_cov2)

plt.axis('off')
plt.tight_layout()
plt.show()




##################
# Figure 5
##################

def draw_plot(data, position, color='k'):

    mean_value = np.mean(data)
    confidence_interval = stats.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=stats.sem(data))
    ci_low, ci_high = confidence_interval
    plt.plot(position, mean_value, 'o', color=color,markersize=10)
    plt.vlines(position, ci_low, ci_high, colors=color, linestyles='solid',linewidth=2)


for hor in range(3):
    df_migra = pd.read_csv('Data/Migra_new.csv', index_col=0, parse_dates=True)
    data = df_migra.iloc[:-6, :]
    std_l_real = []
    mean_real = []

    for col in range(76):
        if not (df_migra.iloc[-6:, col] == 0).all():
            shape = Shape()
            shape.set_shape(df_migra.iloc[-6:, col])
            find = finder(data, shape)
            find.find_patterns(min_d=0.5, select=True)
            pred_ori = find.predict(horizon=3, plot=False, mode='mean')
            std_l_real.append(pred_ori.iloc[:, 2] - pred_ori.iloc[:, 0])
            mean_real.append(len(find.sequences))

    std_l_real = pd.DataFrame(std_l_real)
    mean_real = pd.DataFrame([mean_real, mean_real, mean_real, mean_real, mean_real]).T
    std_l_real = std_l_real * np.sqrt(mean_real) / 1.96

    plt.figure(figsize=(10,10))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#DDDDDD') 
    draw_plot(std_l_real.iloc[:, hor], 0, color='k')


    for k in range(5):
        std_l_fake = []
        mean_fake = []

        for col in range(76):
            if not (df_migra.iloc[-6:, col] == 0).all():
                shape = Shape()
                shape.set_shape(df_migra.iloc[-6:, col])
                find = finder(data, shape)
                find.find_patterns_opposite(min_d=0.5, select=True, dis_sel=k)
                pred_ori = find.predict(horizon=3, plot=False, mode='mean')
                std_l_fake.append(pred_ori.iloc[:, 2] - pred_ori.iloc[:, 0])
                mean_fake.append(len(find.sequences))

        std_l_fake = pd.DataFrame(std_l_fake)
        mean_fake = pd.DataFrame([mean_fake, mean_fake, mean_fake, mean_fake, mean_fake]).T
        std_l_fake = std_l_fake * np.sqrt(mean_fake) / 1.96

        # Tufte-style boxplot for fake data
        draw_plot(std_l_fake.iloc[:, hor], k + 1, color='k') # 3rd quartile line

        # Performing t-tests
        print(ttest_rel(std_l_fake.iloc[:, hor], std_l_real.iloc[:, hor]))
        print(ttest_1samp(np.log((std_l_fake.iloc[:, hor] + 1) / (std_l_real.iloc[:, hor] + 1)), 0))

    plt.xticks([0, 1, 2, 3, 4, 5], ['D<0.5', '0.5>D>0.75','0.75>D>1', '1>D>1.25', '1.25>D>1.5', 'D>1.5'],fontsize=25,rotation=45)
    plt.yticks(fontsize=32)
    if hor==0:
        plt.ylabel('$\sigma$ of Past Futures',fontsize=32)
    plt.title(f'Horizon {hor + 1}',fontsize=40)
    plt.show()



# =============================================================================
# Not in the paper 
# =============================================================================
# best_sd=[]
# # Plotting loop
# for col in range(76):
#     if (not (df_migra.iloc[-9:-3,col] == 0).all()) :
#         shape = Shape()
#         shape.set_shape(df_migra.iloc[-9:-3,col]) 
#         shape_cov1 = Shape()
#         shape_cov1.set_shape(df_fp.iloc[-9:-3,col]) 
#         shape_cov2 = Shape()
#         shape_cov2.set_shape(df_conf.iloc[-9:-3,col]) 
#         shape_cov = [shape_cov1,shape_cov2]
        
#         find = finder_multi(data,cov,shape,shape_cov)
#         find.find_patterns(min_d=3,select=True,metric='dtw')
#         pred_ori = find.predict(horizon=3,plot=False,mode='mean')
#         df_fill = pd.DataFrame([shape.values[-1],shape.values[-1],shape.values[-1]])
#         df_fill= df_fill.T
#         df_fill.columns = pred_ori.columns
#         pred = pd.concat([df_fill,pred_ori],axis=0)
#         pred.index = df_migra.iloc[-4:,col].index
#         pred_ori = pred_ori*(df_migra.iloc[-9:-3,col].max()-df_migra.iloc[-9:-3,col].min())+df_migra.iloc[-9:-3,col].min()
#         te = (df_migra.iloc[-9:-3,col] - df_migra.iloc[-9:-3,col].min()) / (df_migra.iloc[-9:-3,col].max()-df_migra.iloc[-9:-3,col].min())
        
#         std_l_fake = (pred.iloc[:, 2] - pred.iloc[:, 0])* np.sqrt(len(find.sequences)) / 1.96
#         best_sd.append([col,std_l_fake.mean()])

# best_sd=pd.DataFrame(best_sd)
# best_sd=best_sd.sort_values([1])

# fig = plt.figure(figsize=(30, 18))
# num_rows = 7
# num_cols = 9
# width_ratios = [0.75, 2, 0.5]*3
# hei_ratio = [1,0.5]*3+[1]
# gs = GridSpec(num_rows, num_cols, figure=fig, width_ratios=width_ratios,height_ratios=hei_ratio)
# def configure_ax(ax):
#     ax.set(xticks=[], yticks=[])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)


# row=0
# colu=0
# c_tot=0
# for col in best_sd[:9][0]:
#     if (not (df_migra.iloc[-9:-3,col] == 0).all()) & (row<8) :
#         shape = Shape()
#         shape.set_shape(df_migra.iloc[-9:-3,col]) 
#         shape_cov1 = Shape()
#         shape_cov1.set_shape(df_fp.iloc[-9:-3,col]) 
#         shape_cov2 = Shape()
#         shape_cov2.set_shape(df_conf.iloc[-9:-3,col]) 
#         shape_cov = [shape_cov1,shape_cov2]
        
#         find = finder_multi(data,cov,shape,shape_cov)
#         find.find_patterns(min_d=3,select=True,metric='dtw')
#         pred_ori = find.predict(horizon=3,plot=False,mode='mean')
#         df_fill = pd.DataFrame([shape.values[-1],shape.values[-1],shape.values[-1]])
#         df_fill= df_fill.T
#         df_fill.columns = pred_ori.columns
#         pred = pd.concat([df_fill,pred_ori],axis=0)
#         pred.index = df_migra.iloc[-4:,col].index
#         pred_ori = pred_ori*(df_migra.iloc[-9:-3,col].max()-df_migra.iloc[-9:-3,col].min())+df_migra.iloc[-9:-3,col].min()
#         te = (df_migra.iloc[-9:-3,col] - df_migra.iloc[-9:-3,col].min()) / (df_migra.iloc[-9:-3,col].max()-df_migra.iloc[-9:-3,col].min())
        
#         std_l_fake = (pred.iloc[:, 2] - pred.iloc[:, 0])* np.sqrt(len(find.sequences)) / 1.96
#         ax_shape =fig.add_subplot(gs[row, colu+1])
#         ax_shape.plot(te,color='black',linewidth=3)
#         ax_shape.plot(pred.iloc[:,0],color='red',linewidth=3)
#         ax_shape.fill_between(pred.index,pred.iloc[:,0]-std_l_fake,pred.iloc[:,0]+std_l_fake,color='red',alpha=0.3)
#         #ax_shape.set_ylim(-0.2,1.2)
#         ax_shape.grid('y')
#         ax_shape.spines['top'].set_visible(False)
#         ax_shape.spines['right'].set_visible(False)
#         ax_shape.spines['bottom'].set_visible(False)
#         ax_shape.set_xticks([])
#         ax_shape.set_yticks([0,1],['',''])
#         ax_shape.set_title(f'$\sigma$ = {round(best_sd[:9][1].iloc[c_tot],2)}',fontsize=30)
        
#         nested_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[row, colu])
#         ax1 = fig.add_subplot(nested_gs[0, 0])
#         ax1.plot(df_migra.iloc[-9:-3,col],color='red',linewidth=3)
#         configure_ax(ax1)
#         ax2 = fig.add_subplot(nested_gs[1, 0])
#         ax2.plot(df_fp.iloc[-9:-3,col],color='green',linewidth=3)
#         configure_ax(ax2)
#         ax3 = fig.add_subplot(nested_gs[2, 0])
#         ax3.plot(df_conf.iloc[-9:-3,col],color='orange',linewidth=3)
#         configure_ax(ax3)
        
#         colu=colu+3
#         if colu==9:
#             row=row+2
#             colu=0
#         c_tot=c_tot+1
#     else:
#         pass
        
# plt.axis('off')
# plt.tight_layout()
# plt.show()   




# fig = plt.figure(figsize=(30, 18))
# num_rows = 7
# num_cols = 9
# width_ratios = [0.75, 2, 0.5]*3
# hei_ratio = [1,0.5]*3+[1]
# gs = GridSpec(num_rows, num_cols, figure=fig, width_ratios=width_ratios,height_ratios=hei_ratio)
# # Function to configure ax properties
# def configure_ax(ax):
#     ax.set(xticks=[], yticks=[])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)

# # Plotting loop
# row=0
# colu=0
# c_tot=0
# for col in best_sd[-9:][0]:
#     if (not (df_migra.iloc[-9:-3,col] == 0).all()) & (row<8) :
#         shape = Shape()
#         shape.set_shape(df_migra.iloc[-9:-3,col]) 
#         shape_cov1 = Shape()
#         shape_cov1.set_shape(df_fp.iloc[-9:-3,col]) 
#         shape_cov2 = Shape()
#         shape_cov2.set_shape(df_conf.iloc[-9:-3,col]) 
#         shape_cov = [shape_cov1,shape_cov2]
        
#         find = finder_multi(data,cov,shape,shape_cov)
#         find.find_patterns(min_d=3,select=True,metric='dtw')
#         pred_ori = find.predict(horizon=3,plot=False,mode='mean')
#         df_fill = pd.DataFrame([shape.values[-1],shape.values[-1],shape.values[-1]])
#         df_fill= df_fill.T
#         df_fill.columns = pred_ori.columns
#         pred = pd.concat([df_fill,pred_ori],axis=0)
#         pred.index = df_migra.iloc[-4:,col].index
#         pred_ori = pred_ori*(df_migra.iloc[-9:-3,col].max()-df_migra.iloc[-9:-3,col].min())+df_migra.iloc[-9:-3,col].min()
#         te = (df_migra.iloc[-9:-3,col] - df_migra.iloc[-9:-3,col].min()) / (df_migra.iloc[-9:-3,col].max()-df_migra.iloc[-9:-3,col].min())
        
#         std_l_fake = (pred.iloc[:, 2] - pred.iloc[:, 0])* np.sqrt(len(find.sequences)) / 1.96
#         ax_shape =fig.add_subplot(gs[row, colu+1])
#         ax_shape.plot(te,color='black',linewidth=3)
#         ax_shape.plot(pred.iloc[:,0],color='red',linewidth=3)
#         ax_shape.fill_between(pred.index,pred.iloc[:,0]-std_l_fake,pred.iloc[:,0]+std_l_fake,color='red',alpha=0.3)
#         ax_shape.grid('y')
#         ax_shape.spines['top'].set_visible(False)
#         ax_shape.spines['right'].set_visible(False)
#         ax_shape.spines['bottom'].set_visible(False)
#         ax_shape.set_xticks([])
#         ax_shape.set_yticks([0,1],['',''])
#         ax_shape.set_title(f'$\sigma$ = {round(best_sd[-9:][1].iloc[c_tot],2)}',fontsize=30)
        
#         nested_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[row, colu])
#         ax1 = fig.add_subplot(nested_gs[0, 0])
#         ax1.plot(df_migra.iloc[-9:-3,col],color='red',linewidth=3)
#         configure_ax(ax1)
#         ax2 = fig.add_subplot(nested_gs[1, 0])
#         ax2.plot(df_fp.iloc[-9:-3,col],color='green',linewidth=3)
#         configure_ax(ax2)
#         ax3 = fig.add_subplot(nested_gs[2, 0])
#         ax3.plot(df_conf.iloc[-9:-3,col],color='orange',linewidth=3)
#         configure_ax(ax3)
        
#         colu=colu+3
#         if colu==9:
#             row=row+2
#             colu=0
#         c_tot=c_tot+1
#     else:
#         pass
    
# plt.axis('off')
# plt.tight_layout()
# plt.show()   


# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# df_test['plot'] = (df_test['pred']+1)/(df_test['pred']+1)

# fig, ax = plt.subplots(figsize=(15,9))
# world.plot(ax=ax,color='lightgrey', edgecolor='white')
# df_test.plot(ax=ax, column='plot', cmap='Reds_r', vmin=0, vmax=1, legend=False,
#               missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "No migration"})
# plt.tight_layout()
# ax.set_xlim(23,37)
# ax.set_ylim(2.5,12.7)
# ax.set_frame_on(False)
# scalebar = AnchoredSizeBar(ax.transData,
#                            2, '200 km', 'lower left', 
#                            pad=1,
#                            color='black',
#                            frameon=False,
#                            size_vertical=0.1,
#                            fontproperties=fm.FontProperties(size=20))
# ax.annotate('N', xy=(0.95, 0.12), xytext=(0.95, 0.05),
#             arrowprops=dict(facecolor='black', width=0, headwidth=15),
#             xycoords='axes fraction', textcoords='axes fraction',
#             fontsize=40, ha='center')
# ax.add_artist(scalebar)
# ax.text(35,8,'Ethiopia',color='darkred',fontsize=20)
# ax.text(34.7,3.8,'Kenya',color='darkred',fontsize=20)
# ax.text(32,3,'Uganda',color='darkred',fontsize=20)
# ax.text(25,4,'DR Congo',color='darkred',fontsize=20)
# ax.text(24,6,'CAR',color='darkred',fontsize=20)
# ax.text(27,11.5,'Sudan',color='darkred',fontsize=20)
# plt.xticks([])
# plt.yticks([])
# plt.show()








##### MAP (Prediction vs Real)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# # Plot Observed Values
# df_test.plot(column='pred', cmap='Blues', ax=ax1,missing_kwds={
#         "color": "lightgrey",
#         "edgecolor": "red",
#         "hatch": "///",
#         "label": "No migration",
#     })
# ax1.set_title('Predicted Values')
# ax1.set_ylabel('Prediction')

# # Plot Predicted Values
# df_test.plot(column='obs', cmap='Blues', ax=ax2,missing_kwds={
#         "color": "lightgrey",
#         "edgecolor": "red",
#         "hatch": "///",
#         "label": "No migration",
#     })
# ax2.set_title('Observed Values')
# ax2.set_ylabel('Observed')
# plt.tight_layout()
# plt.show()



# mse_lstm=[[],[],[]]
# mse_sf=[[],[],[]]
# mse_ar=[[],[],[]]
# for hor in range(3):
#     for i in range(76):
#         mse_lstm[hor].append(abs(predictions_original_scale[hor,i]-df_migra.iloc[-3+hor,i]))
#         mse_sf[hor].append(abs(pred_tot[i].iloc[hor,0]-df_migra.iloc[-3+hor,i]))
#         mse_ar[hor].append(abs(pred_ar.iloc[hor,i]-df_migra.iloc[-3+hor,i]))
#     model=pd.DataFrame([mse_ar[hor],mse_lstm[hor],mse_sf[hor]]).T
#     model_name=['ARIMA','LSTM','ShapeFinder']
#     ## Map of improvment 
#     df_test= gpd.read_file('D:/Pace_data/Migration/South_Soudan/SS_adm2.shp')

#     best=[]
#     obs_v=[]
#     pred_v=[]
#     for i in range(76):
#         best.append(model_name[model.idxmin(axis=1)[i]])
#         obs_v.append(df_migra.iloc[-3+hor,i])
#         pred_v.append(pred_tot[i].iloc[hor,0])

#     df_m = pd.DataFrame([best,obs_v,pred_v,df_migra.columns]).T

#     df_test = pd.merge(df_test,df_m,left_on='ADM2_EN',right_on=3, how='left')
#     df_test = df_test.rename(columns={0:'best',1:'obs',2:'pred'})

#     # Define custom colormap colors
#     colors = [(0.2078, 0.5922, 1.0),(0.5765, 0.4392, 0.8588),(0.0627, 0.2314, 0.6392)]  # Blue, Green, Red
#     values = [0, 0.5, 1]
#     custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(values, colors)))


#     ##### MAP (best model)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     df_test.plot(column='best', ax=ax, cmap=custom_cmap,legend=True, legend_kwds={'bbox_to_anchor': (1, 1)},missing_kwds={
#             "color": "lightgrey",
#             "edgecolor": "red",
#             "hatch": "///",
#             "label": "No migration",
#         })
#     legend_labels = {category: f"{category} (N={count})" for category, count in df_test['best'].value_counts().items()}
#     legend_labels['No migration']='No migration (N=3)'
#     legend_labels = {
#         'ARIMA': legend_labels['ARIMA'],
#         'LSTM': legend_labels['LSTM'],
#         'ShapeFinder': legend_labels['ShapeFinder'],
#         'No migration': legend_labels['No migration']
#     }
#     handles, labels = ax.get_legend_handles_labels()
#     # Correcting the legend colors
#     legend_colors = {label: colors[i % len(colors)] for i, label in enumerate(legend_labels.keys())}
#     custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=legend_colors[label], markersize=14,
#                                 label=legend_labels[label]) for label in legend_labels]
#     custom_legend[3]=plt.Line2D([0], [0], marker='o', color='w',fillstyle='top', markerfacecolor='lightgrey',markerfacecoloralt='lightgrey',markeredgecolor='red', markersize=14,
#                                 label=legend_labels['No migration'])
#     ax.legend(handles=custom_legend, loc='upper left',fontsize=14)
#     plt.xlabel('Longitude',fontsize=14)
#     plt.ylabel('Latitude',fontsize=14)
#     plt.title(f'Horizon {hor+1}')
#     plt.show()



##################
############## Placebo test 
##################


##################
############## 1
##################

# df_migra = pd.read_csv('D:\Pace_data\Migration\South_Soudan\TS\Migra_new.csv',index_col=0,parse_dates=True)
# data = df_migra.iloc[:-8,:]
# std_l_real=[]
# mean_real=[]
# for col in range(76):
#     if (df_migra.iloc[-8:-3,col]==0).all()==False:
#         shape = Shape()
#         shape.set_shape(df_migra.iloc[-8:-3,col]) 
#         find = finder(data,shape)
#         find.find_patterns(min_d=0.7,select=True)
#         pred_ori = find.predict(horizon=3,plot=False,mode='mean')
#         std_l_real.append(pred_ori.iloc[:,2]-pred_ori.iloc[:,0])
#         mean_real.append(len(find.sequences))
        
# np.random.seed(0)
# shuffled_data = df_migra.values.flatten()
# np.random.shuffle(shuffled_data)
# df_migra_fake = pd.DataFrame(shuffled_data.reshape(df_migra.shape), columns=df_migra.columns)
# data = pd.DataFrame(df_migra_fake).iloc[:-8,:]

# std_l_fake=[]
# mean_fake=[]
# for col in range(76):
#     if (df_migra.iloc[-8:-3,col]==0).all()==False:
#         shape = Shape()
#         shape.set_shape(df_migra.iloc[-8:-3,col]) 
#         find = finder(data,shape)
#         find.find_patterns(min_d=0.7,select=True)
#         pred_ori = find.predict(horizon=3,plot=False,mode='mean')
#         std_l_fake.append(pred_ori.iloc[:,2]-pred_ori.iloc[:,0])
#         mean_fake.append(len(find.sequences))
        
# std_l_fake=pd.DataFrame(std_l_fake)
# std_l_real=pd.DataFrame(std_l_real)
# mean_fake=pd.DataFrame([mean_fake,mean_fake,mean_fake]).T
# mean_real=pd.DataFrame([mean_real,mean_real,mean_real]).T

# std_l_fake=std_l_fake*np.sqrt(mean_fake)/1.96
# std_l_real=std_l_real*np.sqrt(mean_real)/1.96


# plt.boxplot(np.log(std_l_fake.iloc[:,0]/std_l_real.iloc[:,0]),positions=[0])
# plt.boxplot(np.log(std_l_fake.iloc[:,1]/std_l_real.iloc[:,1]),positions=[1])
# plt.boxplot(np.log(std_l_fake.iloc[:,2]/std_l_real.iloc[:,2]),positions=[2])
# plt.hlines(0,-0.5,2.5,linestyle='--')
# plt.title('Log ratio of STD Shuffle data/STD Real data')
# plt.show()

# plt.boxplot(std_l_real.iloc[:,0],positions=[0],showfliers=False)
# plt.boxplot(std_l_fake.iloc[:,0],positions=[1],showfliers=False)
# plt.boxplot(std_l_real.iloc[:,1],positions=[3],showfliers=False)
# plt.boxplot(std_l_fake.iloc[:,1],positions=[4],showfliers=False)
# plt.boxplot(std_l_real.iloc[:,2],positions=[6],showfliers=False)
# plt.boxplot(std_l_fake.iloc[:,2],positions=[7],showfliers=False)
# plt.title('STD Shuffle data/STD Real data')
# plt.xticks([0,1,3,4,6,7],['Real','Fake','Real','Fake','Real','Fake'])
# plt.show()
# ttest_1samp(np.log(std_l_fake.iloc[:,0]/std_l_real.iloc[:,0]),0)





# ##################
# ############## 2
# ##################


# df_migra = pd.read_csv('D:\Pace_data\Migration\South_Soudan\TS\Migra_new.csv',index_col=0,parse_dates=True)
# data = df_migra.iloc[:-5,:]
# std_l_real=[]
# mean_real=[]
# for col in range(76):
#     if (df_migra.iloc[-5:,col]==0).all()==False:
#         shape = Shape()
#         shape.set_shape(df_migra.iloc[-5:,col]) 
#         find = finder(data,shape)
#         find.find_patterns(min_d=0.5,select=True)
#         pred_ori = find.predict(horizon=5,plot=False,mode='mean')
#         std_l_real.append(pred_ori.iloc[:,2]-pred_ori.iloc[:,0])
#         mean_real.append(len(find.sequences))
        

# std_l_fake=[]
# mean_fake=[]
# for col in range(76):
#     if (df_migra.iloc[-5:,col]==0).all()==False:
#         shape = Shape()
#         shape.set_shape(df_migra.iloc[-5:,col]) 
#         find = finder(data,shape)
#         find.find_patterns_opposite(min_d=0.5,select=True)
#         pred_ori = find.predict(horizon=5,plot=False,mode='mean')
#         std_l_fake.append(pred_ori.iloc[:,2]-pred_ori.iloc[:,0])
#         mean_fake.append(len(find.sequences))
        

# std_l_fake=pd.DataFrame(std_l_fake)
# std_l_real=pd.DataFrame(std_l_real)
# mean_fake=pd.DataFrame([mean_fake,mean_fake,mean_fake,mean_fake,mean_fake]).T
# mean_real=pd.DataFrame([mean_real,mean_real,mean_real,mean_real,mean_real]).T

# std_l_fake=std_l_fake*np.sqrt(mean_fake)/1.96
# std_l_real=std_l_real*np.sqrt(mean_real)/1.96

# plt.figure(figsize=(15,10))
# plt.boxplot(std_l_real.iloc[:,0],positions=[0],showfliers=False)
# plt.boxplot(std_l_fake.iloc[:,0],positions=[1],showfliers=False)
# plt.boxplot(std_l_real.iloc[:,1],positions=[3],showfliers=False)
# plt.boxplot(std_l_fake.iloc[:,1],positions=[4],showfliers=False)
# plt.boxplot(std_l_real.iloc[:,2],positions=[6],showfliers=False)
# plt.boxplot(std_l_fake.iloc[:,2],positions=[7],showfliers=False)
# plt.boxplot(std_l_real.iloc[:,3],positions=[9],showfliers=False)
# plt.boxplot(std_l_fake.iloc[:,3],positions=[10],showfliers=False)
# plt.boxplot(std_l_real.iloc[:,4],positions=[12],showfliers=False)
# plt.boxplot(std_l_fake.iloc[:,4],positions=[13],showfliers=False)
# plt.title('STD Shuffle data/STD Real data')
# plt.xticks([0,1,3,4,6,7,9,10,12,13],['Min_d_tail','Max_dist_tail']*5)
# plt.show()

# print(ttest_rel(std_l_fake.iloc[:,0],std_l_real.iloc[:,0]))
# print(ttest_rel(std_l_fake.iloc[:,1],std_l_real.iloc[:,1]))
# print(ttest_rel(std_l_fake.iloc[:,2],std_l_real.iloc[:,2]))
# print(ttest_rel(std_l_fake.iloc[:,3],std_l_real.iloc[:,3]))
# print(ttest_rel(std_l_fake.iloc[:,4],std_l_real.iloc[:,4]))

# print(ttest_1samp(np.log((std_l_fake.iloc[:,0]+1)/(std_l_real.iloc[:,0]+1)),0))
# print(ttest_1samp(np.log((std_l_fake.iloc[:,1]+1)/(std_l_real.iloc[:,1]+1)),0))
# print(ttest_1samp(np.log((std_l_fake.iloc[:,2]+1)/(std_l_real.iloc[:,2]+1)),0))
# print(ttest_1samp(np.log((std_l_fake.iloc[:,3]+1)/(std_l_real.iloc[:,3]+1)),0))
# print(ttest_1samp(np.log((std_l_fake.iloc[:,4]+1)/(std_l_real.iloc[:,4]+1)),0))


# plt.boxplot(np.log(std_l_fake.iloc[:,0]/std_l_real.iloc[:,0]),positions=[0])
# plt.boxplot(np.log(std_l_fake.iloc[:,1]/std_l_real.iloc[:,1]),positions=[1])
# plt.boxplot(np.log(std_l_fake.iloc[:,2]/std_l_real.iloc[:,2]),positions=[2])
# plt.boxplot(np.log(std_l_fake.iloc[:,3]/std_l_real.iloc[:,3]),positions=[3])
# plt.boxplot(np.log(std_l_fake.iloc[:,4]/std_l_real.iloc[:,4]),positions=[4])
# plt.hlines(0,-0.5,4.5,linestyle='--')
# plt.title('Log ratio of STD Max distance tail/STD Min distance tail')
# plt.show()



# gdf_shapefinder = df_test[df_test['best'] == 'ShapeFinder'][['geometry','obs']]
# gdf_arima_lstm = df_test[(df_test['best'] == 'ARIMA') | (df_test['best'] == 'LSTM')][['geometry','obs']]
# max_val = max(df_test['obs'].max(), gdf_arima_lstm['obs'].max(), gdf_shapefinder['obs'].max())
# gdf_shapefinder['obs']=(gdf_shapefinder['obs']/max_val)
# gdf_arima_lstm['obs']=(gdf_arima_lstm['obs']/max_val)
# gdf_shapefinder['obs']=gdf_shapefinder['obs'].astype(float)
# gdf_arima_lstm['obs']=gdf_arima_lstm['obs'].astype(float)
# colors1 = [(0.8,0.8,1),(0, 0, 0.6)]
# colors2 = [(1,0.8,0.8),(0.6, 0, 0)]
# values = [0, 1]
# custom_cmap_1 = LinearSegmentedColormap.from_list('custom_cmap_1', list(zip(values, colors1)))
# custom_cmap_2 = LinearSegmentedColormap.from_list('custom_cmap_2', list(zip(values, colors2)))
# fig, ax = plt.subplots(figsize=(10, 6))
# gdf_shapefinder.plot('obs', cmap=custom_cmap_1,ax=ax,vmax=1, vmin=0, edgecolor='grey')
# gdf_arima_lstm.plot('obs', cmap=custom_cmap_2,ax=ax,vmax=1, vmin=0, edgecolor='grey')
# ax.set_title('Observations - ShapeFinder (Blue) and ARIMA/LSTM (Red)')
# plt.show()
