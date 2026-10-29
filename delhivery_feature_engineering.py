# --- Cell 4 ---import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import re
# --- Cell 6 ---df=pd.read_csv('/kaggle/input/delhivery/delhivery_data.csv')
df.head(20)
# --- Cell 7 ---df[['route_schedule_uuid','route_type','trip_uuid','source_center','source_name',
   'destination_center','destination_name','od_start_time','od_end_time',
   'start_scan_to_end_scan','is_cutoff','actual_distance_to_destination','actual_time','segment_osrm_time','segment_osrm_distance']].head(20)
# --- Cell 8 ---sample=df[(df['route_schedule_uuid']=='thanos::sroute:eb7bfc78-b351-4c0e-a951-fa3d5c3297ef') & (df['trip_uuid']=='trip-153741093647649320')]
sample[['source_name','destination_name','od_start_time','od_start_time','od_end_time','start_scan_to_end_scan','is_cutoff','actual_distance_to_destination','actual_time','segment_osrm_time','segment_osrm_distance']]

# --- Cell 9 ---df.shape
# --- Cell 10 ---df.dtypes
# --- Cell 11 ---df.info()
# --- Cell 12 ---df.describe(include=object)
# --- Cell 13 ---df.head()
# --- Cell 14 ---df.dtypes
# --- Cell 16 ---df['trip_creation_time']=pd.to_datetime(df['trip_creation_time'])
df['od_start_time']=pd.to_datetime(df['od_start_time'])
df['od_end_time']=pd.to_datetime(df['od_end_time'])
df['cutoff_timestamp']=pd.to_datetime(df['cutoff_timestamp'], format='mixed')



# --- Cell 17 ---df.dtypes
# --- Cell 18 ---df.head()
# --- Cell 20 ---df.isna().sum()
# --- Cell 22 ---# drop missing values 
df.drop(df[df['source_name'].isna() | df['destination_name'].isna()].index , inplace=True)
# --- Cell 23 ---df.isna().sum()
# --- Cell 25 ---df['segment_key']=df.apply(lambda x:"#".join([x['trip_uuid'],x['source_center'],x['destination_center']]), axis=1)
# --- Cell 26 ---# sample=df[(df['route_schedule_uuid']=='thanos::sroute:eb7bfc78-b351-4c0e-a951-fa3d5c3297ef') & (df['trip_uuid']=='trip-153741093647649320')]
# sample[['source_name','destination_name','od_start_time','od_start_time','od_end_time','start_scan_to_end_scan','is_cutoff','actual_distance_to_destination','actual_time','segment_osrm_time','segment_osrm_distance']]
# sample.head(20)
# sample.columns
# --- Cell 27 ---df['segment_actual_time_cumsum']=df.groupby('segment_key')['segment_actual_time'].transform(lambda x:x.cumsum())

df['segment_osrm_time_cumsum']=df.groupby('segment_key')['segment_osrm_time'].transform(lambda x:x.cumsum())

df['segment_osrm_distance_cumsum']=df.groupby('segment_key')['segment_osrm_distance'].transform(lambda x:x.cumsum())
# --- Cell 28 ---# df.groupby('segment_key')['segment_actual_time_sum'].nth(-1)
# --- Cell 29 ---df['segment_actual_time_sum']=df.groupby('segment_key')['segment_actual_time_cumsum'].transform(lambda x:x.iloc[-1])
df['segment_osrm_time_sum']=df.groupby('segment_key')['segment_osrm_time_cumsum'].transform(lambda x:x.iloc[-1])
df['segment_osrm_distance_sum']=df.groupby('segment_key')['segment_osrm_distance_cumsum'].transform(lambda x:x.iloc[-1])

# --- Cell 30 ---df=df.sort_values(by=['segment_key','od_end_time'])
df
# --- Cell 31 ---# s1=df[df['segment_key']=='trip-153741093647649320#IND388121AAA#IND388620AAB']
# s1[['segment_key','od_end_time','segment_actual_time','segment_osrm_time', 'segment_osrm_distance',
#    'segment_actual_time_sum', 'segment_osrm_time_sum',
#        'segment_osrm_distance_sum', 'segment_actual_time_cumsum',
#        'segment_osrm_time_cumsum', 'segment_osrm_distance_cumsum']]
# # s1.columns
# --- Cell 33 ---df['od_time_diff_hour']=(df['od_end_time']-df['od_start_time'])/pd.Timedelta(hours=1)
# --- Cell 35 ---df.columns
# --- Cell 36 ---df['trip_creation_year']=df['trip_creation_time'].dt.year 
df['trip_creation_month']=df['trip_creation_time'].dt.month 
df['trip_creation_day']=df['trip_creation_time'].dt.day

# --- Cell 37 ---def get_state(name):
    pattern="\([A-Za-z]+\s?\w+\)"
    pattern="\([A-Za-z &]+\s?\w+\)"
    state=re.findall(pattern, name)[0]
    state=state.replace("(","")
    state=state.replace(")","")
    return state
# --- Cell 38 ---def get_city(name):
    pattern="\([A-Za-z &]+\s?\w+\)"
    state=re.findall(pattern, name)[0]
    city_place_code=name.replace(state,'')
    city_place_code_parts=city_place_code.split("_")
    if len(city_place_code_parts)==1 :
        city=city_place_code_parts[0].strip()
    elif len(city_place_code_parts)==2:
        city=city_place_code.strip()
    elif len(city_place_code_parts)==3 or len(city_place_code_parts)==4:
        city=city_place_code_parts[0].strip()
    else:
        city=city_place_code
    return city 
# --- Cell 39 ---def get_place(name):
    pattern="\([A-Za-z &]+\s?\w+\)"
    try:
        state=re.findall(pattern, name)[0]
        city_place_code=name.replace(state,'')
        city_place_code_parts=city_place_code.split("_")
        if len(city_place_code_parts)==3 or len(city_place_code_parts)==4:
            place=city_place_code_parts[1].strip()
        else:
            place=None
        return place 
    except Exception as exp:
        return None

# --- Cell 40 ---def get_code(name):
    pattern="\([A-Za-z &]+\s?\w+\)"
    try:
        state=re.findall(pattern, name)[0]
        city_place_code=name.replace(state,'')
        city_place_code_parts=city_place_code.split("_")
        if len(city_place_code_parts)==3 :
            code=city_place_code_parts[2].strip()
        elif  len(city_place_code_parts)==4:
            code="_".join(city_place_code_parts[2:]).strip()
        else:
            code=None
        return code 
    except Exception as exp:
        return None

# --- Cell 41 ---df['destination_state']=df['destination_name'].map(get_state)
df['source_state']=df['source_name'].map(get_state)
# --- Cell 42 ---df['destination_city']=df['destination_name'].map(get_city)
df['source_city']=df['source_name'].map(get_state)

# --- Cell 43 ---df['destination_place']=df['destination_name'].map(get_place)
df['source_place']=df['source_name'].map(get_place)
# --- Cell 44 ---df['destination_code']=df['destination_name'].map(get_code)
df['source_code']=df['source_name'].map(get_code)
# --- Cell 46 ---# df2=df[['trip_uuid','segment_key','actual_distance_to_destination', 'actual_time','osrm_time', 'osrm_distance',
#    'segment_actual_time','segment_osrm_time', 'segment_osrm_distance',
#    'segment_actual_time_cumsum', 'segment_osrm_time_cumsum',
#         'segment_osrm_distance_cumsum', 'segment_actual_time_sum',
#         'segment_osrm_time_sum', 'segment_osrm_distance_sum','od_time_diff_hour','od_start_time','od_end_time']]
# --- Cell 47 ---# df2=df2[df2['trip_uuid']=='trip-153741093647649320']
# df2
# --- Cell 49 ---trip_actual_time_sum_df=df.groupby(['trip_uuid','segment_key']).nth(-1).groupby(['trip_uuid']).agg(trip_actual_time_sum=('actual_time','sum')).reset_index()
trip_actual_time_sum_df
# --- Cell 50 ---trip_segment_actual_time_sum_df=df.groupby(['trip_uuid','segment_key']).nth(-1).groupby(['trip_uuid']).agg(trip_segment_actual_time_sum=('segment_actual_time_sum','sum')).reset_index()
trip_segment_actual_time_sum_df
# --- Cell 51 ---trip_segment_osrm_time_sum_df=df.groupby(['trip_uuid','segment_key']).nth(-1).groupby(['trip_uuid']).agg(trip_segment_osrm_time_sum=('segment_osrm_time_sum','sum')).reset_index()
trip_segment_osrm_time_sum_df
# --- Cell 52 ---trip_segment_osrm_distance_sum_df=df.groupby(['trip_uuid','segment_key']).nth(-1).groupby(['trip_uuid']).agg(trip_segment_osrm_distance_sum=('segment_osrm_distance_sum','sum')).reset_index()
trip_segment_osrm_distance_sum_df
# --- Cell 53 ---trip_osrm_time_sum_df=df.groupby(['trip_uuid','segment_key']).nth(-1).groupby(['trip_uuid']).agg(trip_osrm_time_sum=('osrm_time','sum')).reset_index()
trip_osrm_time_sum_df
# --- Cell 54 ---trip_osrm_distance_sum_df=df.groupby(['trip_uuid','segment_key']).nth(-1).groupby(['trip_uuid']).agg(trip_osrm_distance_sum=('osrm_distance','sum')).reset_index()
trip_osrm_distance_sum_df
# --- Cell 55 ---trip_total_time_in_hrs_df=df.groupby(['trip_uuid','segment_key']).nth(-1).groupby(['trip_uuid']).agg(trip_total_time_in_hrs=('od_time_diff_hour','sum')).reset_index()
trip_total_time_in_hrs_df
# --- Cell 57 ---trip_aggregated_df=pd.concat(
    objs=(
            iDF.set_index('trip_uuid') for iDF in (trip_actual_time_sum_df,trip_osrm_time_sum_df, trip_osrm_distance_sum_df, 
                                                 trip_segment_actual_time_sum_df,trip_segment_osrm_time_sum_df,trip_segment_osrm_distance_sum_df,
                                                 trip_total_time_in_hrs_df)
         ),
    axis=1, 
    join='inner'
).reset_index()
trip_aggregated_df
# --- Cell 58 ---
df=pd.merge(df,trip_aggregated_df, on='trip_uuid' )
df
# --- Cell 59 ---# df['trip_acutal_time_sum']=df['actual_time'].map(groupby(['trip_uuid','segment_key']).nth(-1).groupby(['trip_uuid']).agg({'actual_time':np.sum})
# --- Cell 61 ---trip_aggregated_df
# --- Cell 62 ---
import warnings
warnings.filterwarnings("ignore")
fig, axs = plt.subplots(3, 2,  figsize=(10,10))
sns.histplot(ax=axs[0,0],data= trip_aggregated_df['trip_actual_time_sum'],kde=True)
sns.boxplot(ax=axs[0,1],data=trip_aggregated_df['trip_actual_time_sum'])

sns.histplot(ax=axs[1,0],data= trip_aggregated_df['trip_osrm_time_sum'],kde=True)
sns.boxplot(ax=axs[1,1],data= trip_aggregated_df['trip_osrm_time_sum'])

sns.histplot(ax=axs[2,0],data= trip_aggregated_df['trip_osrm_distance_sum'],kde=True)
sns.boxplot(ax=axs[2,1],data= trip_aggregated_df['trip_osrm_distance_sum'])


plt.show()


# --- Cell 63 ---import warnings
warnings.filterwarnings("ignore")
fig, axs = plt.subplots(3, 2,  figsize=(10,10))

sns.histplot(ax=axs[0,0],data= trip_aggregated_df['trip_segment_actual_time_sum'],kde=True)
sns.boxplot(ax=axs[0,1],data= trip_aggregated_df['trip_segment_actual_time_sum'])


sns.histplot(ax=axs[1,0],data= trip_aggregated_df['trip_segment_osrm_time_sum'],kde=True)
sns.boxplot(ax=axs[1,1],data= trip_aggregated_df['trip_segment_osrm_time_sum'])

sns.histplot(ax=axs[2,0],data= trip_aggregated_df['trip_segment_osrm_distance_sum'],kde=True)
sns.boxplot(ax=axs[2,1],data= trip_aggregated_df['trip_segment_osrm_distance_sum'])


plt.show()
# --- Cell 64 ---import warnings
warnings.filterwarnings("ignore")
fig, axs = plt.subplots(ncols= 2,  figsize=(10,5))
sns.histplot(ax=axs[0],data= trip_aggregated_df['trip_total_time_in_hrs'],kde=True)
sns.boxplot(ax=axs[1],data= trip_aggregated_df['trip_total_time_in_hrs'])
plt.show()

# --- Cell 65 ---sns.histplot(data=trip_aggregated_df['trip_actual_time_sum'],kde=True)
# --- Cell 66 ---sns.boxplot(data=trip_aggregated_df['trip_actual_time_sum'])
# --- Cell 68 ---def clip_value_helper(row,cl, Q1,Q3, minval, maxval):
    #     Q1=row[cl].quantile(0.25)
    #     Q3=row[cl].quantile(0.75)
    #     minval=min(row[cl])
    #     maxval=max(row[cl])
    IQR=Q3-Q1
    if row[cl]<Q1-1.5*IQR:
        return min(minval, Q1-1.5*IQR)
    elif row[cl] > Q3+1.5*IQR:
        return min(maxval, Q3+1.5*IQR)
    else:
        return row[cl]
   
     
# --- Cell 69 ---trip_aggregated_df.columns
# --- Cell 70 ---for cl in ['trip_actual_time_sum', 'trip_osrm_time_sum', 'trip_osrm_distance_sum', 'trip_segment_actual_time_sum','trip_segment_osrm_time_sum', 'trip_segment_osrm_distance_sum','trip_total_time_in_hrs']:
    Q1=trip_aggregated_df[cl].quantile(0.25)
    Q3=trip_aggregated_df[cl].quantile(0.75)
    minval=min(trip_aggregated_df[cl])
    maxval=max(trip_aggregated_df[cl])
    trip_aggregated_df[cl]=trip_aggregated_df.apply(lambda row:clip_value_helper(row,cl,
                                                                     Q1,Q3,minval, maxval) , axis=1)

# --- Cell 71 ---trip_aggregated_df
# --- Cell 73 ---
import warnings
warnings.filterwarnings("ignore")
fig, axs = plt.subplots(3, 2,  figsize=(10,10))
sns.histplot(ax=axs[0,0],data= trip_aggregated_df['trip_actual_time_sum'],kde=True)
sns.boxplot(ax=axs[0,1],data=trip_aggregated_df['trip_actual_time_sum'])

sns.histplot(ax=axs[1,0],data= trip_aggregated_df['trip_osrm_time_sum'],kde=True)
sns.boxplot(ax=axs[1,1],data= trip_aggregated_df['trip_osrm_time_sum'])

sns.histplot(ax=axs[2,0],data= trip_aggregated_df['trip_osrm_distance_sum'],kde=True)
sns.boxplot(ax=axs[2,1],data= trip_aggregated_df['trip_osrm_distance_sum'])


plt.show()


# --- Cell 74 ---import warnings
warnings.filterwarnings("ignore")
fig, axs = plt.subplots(3, 2,  figsize=(10,10))

sns.histplot(ax=axs[0,0],data= trip_aggregated_df['trip_segment_actual_time_sum'],kde=True)
sns.boxplot(ax=axs[0,1],data= trip_aggregated_df['trip_segment_actual_time_sum'])


sns.histplot(ax=axs[1,0],data= trip_aggregated_df['trip_segment_osrm_time_sum'],kde=True)
sns.boxplot(ax=axs[1,1],data= trip_aggregated_df['trip_segment_osrm_time_sum'])

sns.histplot(ax=axs[2,0],data= trip_aggregated_df['trip_segment_osrm_distance_sum'],kde=True)
sns.boxplot(ax=axs[2,1],data= trip_aggregated_df['trip_segment_osrm_distance_sum'])


plt.show()
# --- Cell 75 ---import warnings
warnings.filterwarnings("ignore")
fig, axs = plt.subplots(ncols= 2,  figsize=(10,5))
sns.histplot(ax=axs[0],data= trip_aggregated_df['trip_total_time_in_hrs'],kde=True)
sns.boxplot(ax=axs[1],data= trip_aggregated_df['trip_total_time_in_hrs'])
plt.show()

# --- Cell 78 ---from scipy.stats import ttest_rel, ttest_ind 

# --- Cell 79 ---stat, pval=ttest_rel(trip_aggregated_df['trip_actual_time_sum'],
                     trip_aggregated_df['trip_osrm_time_sum'],
                    alternative='greater')
print(f"stat {stat} pval {pval}")
# --- Cell 80 ---if pval <0.05:
    print("We will reject H0 :")
    print("mean Actual time to deliver package from source to destination is greater than OSRM time")
else:
    print("We fail to  reject H0 :")
    print("mean Actual time to deliver package from source to destination is lesser or equal  than OSRM time")
# --- Cell 81 ---np.mean(trip_aggregated_df['trip_actual_time_sum']), np.mean(trip_aggregated_df['trip_osrm_time_sum'])
# --- Cell 83 ---stat, pval=ttest_rel(trip_aggregated_df['trip_actual_time_sum'],
                     trip_aggregated_df['trip_segment_actual_time_sum'],
                    alternative='greater')
print(f"stat {stat} pval {pval}")
# --- Cell 84 ---if pval <0.05:
    print("We will reject H0 :")
    print("mean Actual trip aggregated time   to deliver package from source to destination is greater than segment actual aggregated time")
else:
    print("We fail to  reject H0 :")
    print("mean Actual trip aggregated time to deliver package from source to destination is lesser or equal  than segment actual aggregated time")
# --- Cell 86 ---stat, pval=ttest_rel(trip_aggregated_df['trip_osrm_distance_sum'],
                     trip_aggregated_df['trip_segment_osrm_distance_sum'],
                    alternative='greater')
print(f"stat {stat} pval {pval}")
# --- Cell 87 ---if pval <0.05:
    print("We will reject H0 :")
    print("mean Actual trip OSRM distance for trip  to deliver package from source to destination is greater than segment actual aggregated OSRM distance for trip")
else:
    print("We fail to  reject H0 :")
    print("mean Actual trip OSRM distance for trip to deliver package from source to destination is lesser or equal  than segment actual aggregated OSRM distance for trip")
# --- Cell 88 ---trip_aggregated_df
# --- Cell 89 ---np.max(trip_aggregated_df['trip_actual_time_sum']), np.max(trip_aggregated_df['trip_segment_actual_time_sum'])
# --- Cell 91 ---stat, pval=ttest_rel(trip_aggregated_df['trip_osrm_time_sum'],
                     trip_aggregated_df['trip_segment_osrm_time_sum'],
                    alternative='greater')
print(f"stat {stat} pval {pval}")
# --- Cell 92 ---if pval <0.05:
    print("We will reject H0 :")
    print("mean Actual trip aggregated  OSRM time  for trip  to deliver package from source to destination is greater than segment actual aggregated OSRM time for trip")
else:
    print("We fail to  reject H0 :")
    print("mean Actual trip aggregated OSRM time  for trip to deliver package from source to destination is lesser or equal  than segment actual aggregated OSRM time for trip")
# --- Cell 95 ---# Check from where most orders are coming from (State, Corridor, etc.)


orders_from_df=df.groupby('source_state').agg(count_trips=('trip_uuid','nunique')).reset_index()
orders_from_df=orders_from_df.sort_values(by='count_trips', ascending=False)
orders_from_df
# --- Cell 96 ---sns.barplot(data=orders_from_df,x='source_state',y='count_trips')
plt.xticks(rotation=90)
plt.show()
# --- Cell 98 ---orders_to_df=df.groupby('destination_state').agg(count_trips=('trip_uuid','nunique')).reset_index()
orders_to_df=orders_to_df.sort_values(by='count_trips', ascending=False)
orders_to_df
# --- Cell 99 ---sns.barplot(data=orders_to_df,x='destination_state',y='count_trips')
plt.xticks(rotation=90)
plt.show()
# --- Cell 101 ---# corriodor_df=df[['source_center','destination_center','actual_time','osrm_time', 'osrm_distance']]

df['corridor']=df.apply(lambda x:"#".join([x['source_center'],x['destination_center']]), axis=1)
df
# --- Cell 102 ---corridor_total_trips=df.groupby(['trip_uuid','corridor']).nth(-1).groupby(['corridor']).agg(total_trips=('trip_uuid','nunique')).reset_index()
corridor_total_trips
# --- Cell 103 ---corridor_actual_time_mean_df=df.groupby(['trip_uuid','corridor']).nth(-1).groupby(['corridor']).agg(corridor_actual_time_mean=('segment_actual_time_cumsum','mean')).reset_index()
corridor_actual_time_mean_df
# --- Cell 104 ---corridor_osrm_time_mean_df=df.groupby(['trip_uuid','corridor']).nth(-1).groupby(['corridor']).agg(corridor_dsrm_time_mean=('segment_osrm_time_cumsum','mean')).reset_index()
corridor_osrm_time_mean_df
# --- Cell 105 ---corridor_osrm_distance_mean_df=df.groupby(['trip_uuid','corridor']).nth(-1).groupby(['corridor']).agg(corridor_osrm_distance_mean=('segment_osrm_distance_cumsum','mean')).reset_index()
corridor_osrm_distance_mean_df
# --- Cell 106 ---corridor_aggregated_df=pd.concat(
    objs=(
            iDF.set_index('corridor') for iDF in (corridor_total_trips,
                                                  corridor_actual_time_mean_df,
                                                  corridor_osrm_distance_mean_df,
                                                   corridor_osrm_time_mean_df
                                                 )
         ),
    axis=1, 
    join='inner'
).reset_index()
corridor_aggregated_df
# --- Cell 108 ---top_20Trips_corridor_df=corridor_aggregated_df[['corridor','total_trips']].sort_values(by='total_trips', ascending=False)[:20]
top_20Trips_corridor_df
# --- Cell 109 ---sns.barplot(data=top_20Trips_corridor_df,x='corridor',y='total_trips')
plt.xticks(rotation=90)
plt.show()
# --- Cell 111 ---top_20trip_time_corridor_df=corridor_aggregated_df[['corridor','corridor_actual_time_mean']].sort_values(by='corridor_actual_time_mean', ascending=False)[:20]
top_20trip_time_corridor_df
# --- Cell 112 ---sns.barplot(data=top_20trip_time_corridor_df,x='corridor',y='corridor_actual_time_mean')
plt.xticks(rotation=90)
plt.show()
# --- Cell 113 ---corridor_aggregated_df[['corridor_actual_time_mean','corridor_osrm_distance_mean','corridor_dsrm_time_mean','total_trips']].corr()
# --- Cell 115 ---top_20trip_dist_corridor_df=corridor_aggregated_df[['corridor','corridor_osrm_distance_mean']].sort_values(by='corridor_osrm_distance_mean', ascending=False)[:20]
top_20trip_dist_corridor_df
# --- Cell 116 ---sns.barplot(data=top_20trip_dist_corridor_df,x='corridor',y='corridor_osrm_distance_mean')
plt.xticks(rotation=90)
plt.show()
