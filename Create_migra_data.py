# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:23:27 2023

@author: thoma
"""

import pandas as pd 
import os 
import geopandas as gpd
from shapely import wkt

########################################################################################################################################################################################
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
#                                   Preprocess of the data                                                                                                                             #
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
########################################################################################################################################################################################

# =============================================================================
# Migra data
# =============================================================================

ind_all=pd.date_range(start='01/01/2020',end='31/10/2022',freq='M')
count=0
for elmt in os.listdir('Data/Migration/'):
    if count == 24:
        df=pd.read_excel('Data/Migration/'+elmt,parse_dates=True)
        for i in range(1,11):
            sub = df[df['interview.month'].dt.month==i]
            df_real=sub[sub['dep.country']== 'SSD']
            df_group=df_real.groupby(by=['dep.adm2','dest.adm1'])['total.ind'].sum()
            if count == 0: 
                df_tot=pd.DataFrame(df_group)
                df_tot.columns=[str(ind_all[0])[0:7]]
            else: 
                df_tot[str(ind_all[count])[0:7]]=df_group
            count=count+1    
    else :     
        df=pd.read_excel('Data/Migration/'+elmt)
        df_real=df[df['dep.country']== 'SSD']
        df_group=df_real.groupby(by=['dep.adm2','dest.adm1'])['total.ind'].sum()
        if count == 0: 
            df_tot=pd.DataFrame(df_group)
            df_tot.columns=[str(ind_all[0])[0:7]]
        else: 
            df_tot[str(ind_all[count])[0:7]]=df_group
    
    count=count+1    

df_tot=df_tot.fillna(0)   
df_tot_2=df_tot.reset_index(level=[0,1]) 
j=df_tot_2.iloc[:,0:2].values.tolist()
df_tot_2=df_tot_2.iloc[:,2:]
df_tot_2=df_tot_2.transpose()
l_col=[]
for elmt in j:
    l_col.append(elmt[0]+' / '+elmt[1])
df_tot_2.columns=l_col
df_tot_2.index=ind_all
del df_real,df_group,elmt,l_col,count


dep_u=[]
dest_u=[]
for elmt in j:
    dep_u.append(elmt[0])
    dest_u.append(elmt[1])
dest_u = list(set(dest_u))
dep_u =  list(set(dep_u))

df_tot_3=df_tot.reset_index(level=[0,1]) 
df_tot_3=df_tot_3.drop('dest.adm1',axis=1)
df_tot_3=df_tot_3.groupby(by='dep.adm2').sum()
df_tot_3=df_tot_3.transpose()
df_tot_3.index=ind_all
ind_mig=pd.date_range(start='01/01/2020',end='28/02/2022',freq='M')
df_mig=df_tot_3
df_mig=df_mig.rename(columns={'Abyei Area':'Abyei Region', 'Canal/Pigi': 'Canal-Pigi','Kajo-Keji':'Kajo-keji'})
df_mig=df_mig.drop(columns=['Unknown'])
df_mig.to_csv('Data/Migra_new.csv')

# =============================================================================
# FOOD price
# =============================================================================

df_fp = pd.read_csv('Data/Food_price_2.csv',parse_dates=True)
df_fp=df_fp[df_fp.mkt_name != 'Market Average']
df_fp_beans = df_fp.pivot(index='DATES', columns='mkt_name', values='o_beans')

market_loc = pd.DataFrame([df_fp.mkt_name.unique(),df_fp.lat.unique(),df_fp.lon.unique()])
market_loc=market_loc.T
market_loc = gpd.GeoDataFrame(market_loc,geometry=gpd.points_from_xy(market_loc.iloc[:,2],market_loc.iloc[:,1]))

df_test= gpd.read_file('D:/Pace_data/Migration/South_Soudan/SS_adm2.shp')
df_test['nearest_market'] = df_test['geometry'].apply(lambda x: market_loc.loc[market_loc.distance(x).idxmin(), 0])

df_food = pd.DataFrame(index = df_fp_beans.index, columns=df_test.ADM2_EN)
for i in range(len(df_test['nearest_market'])):
    df_food.iloc[:,i] = df_fp_beans.loc[:,df_test['nearest_market'][i]]

df_food = df_food[df_mig.columns]
df_food.to_csv('Data/Food.csv')



# =============================================================================
# Conflict 
# =============================================================================

df_migra=pd.read_csv('Data/Migra_new.csv',index_col=0,parse_dates=(True))

df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged231-csv.zip",
                 parse_dates=['date_start','date_end'],low_memory=False)
df= df[df.country=='South Sudan']
df['geometry'] = df['geom_wkt'].apply(wkt.loads)
gdf_points = gpd.GeoDataFrame(df, geometry='geometry')

df_tot= pd.DataFrame(columns=df_test.ADM2_EN,index=pd.date_range(df.date_start.min(),
                                          df.date_end.max()))


df_tot=df_tot.fillna(0)
c=0
for i in df_test.geometry:
    polygon = gpd.GeoSeries(i)
    df_sub=gpd.sjoin(gdf_points, gpd.GeoDataFrame(geometry=polygon), how='inner', op='within')
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j] == df_sub.date_end.iloc[j]:
            df_tot.loc[df_sub.date_start.iloc[j],df_test.ADM2_EN[c]]=df_tot.loc[df_sub.date_start.iloc[j],df_test.ADM2_EN[c]]+df_sub.best.iloc[j]
        else:
            df_tot.loc[df_sub.date_start.iloc[j]:
            df_sub.date_end.iloc[j],df_test.ADM2_EN[c]]=df_tot.loc[df_sub.date_start.iloc[j]: \
                                                  df_sub.date_end.iloc[j],df_test.ADM2_EN[c]]+ \
                                                  df_sub.best.iloc[j]/ \
                                                  (df_sub.date_end.iloc[j]- \
                                                  df_sub.date_start.iloc[j]).days 
    c += 1
df_tot=df_tot[df_migra.columns]
df_tot=df_tot.resample('M').sum()
df_tot.to_csv('Data/Conf.csv')

