#Imported libraries and modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mlflow


#loading dataset
df=pd.read_csv('/workspace/data/Dataset.txt', delimiter=';',low_memory=False)

#data_preprocessing
df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],dayfirst=True)
cols_to_process=df.select_dtypes(include='object').columns.difference(["Date","Time"])
df[cols_to_process]=df[cols_to_process].apply(
    lambda col: pd.to_numeric(col,errors='coerce')
)
df = df.dropna(subset=cols_to_process)  
df['date_time']=pd.to_datetime(df['date_time']) 
df['year'] = df['date_time'].apply(lambda x: x.year)
df['quarter'] = df['date_time'].apply(lambda x: x.quarter)
df['month'] = df['date_time'].apply(lambda x: x.month)
df['day'] = df['date_time'].apply(lambda x: x.day)
df=df.loc[:,['date_time','Global_active_power','Global_intensity' ,'Sub_metering_1','Sub_metering_2','Sub_metering_3','year','quarter','month','day']]
df.sort_values('date_time', inplace=True, ascending=True)
df = df.reset_index(drop=True)
df["weekday"]=df.apply(lambda row: row["date_time"].weekday(),axis=1)
df["weekday"] = (df["weekday"] <    5).astype(int)

dataset = df.select_dtypes(include='number').values #numpy.ndarray
dataset = dataset.astype('float32')
dataset = np.log1p(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
#-----------------------------------------------------------------------------------
series = dataset.astype("float32")
split = int(0.8 * len(series))
train_series = series[:split]
test_series  = series[split:]
train_series = scaler.fit_transform(train_series)
test_series =scaler.transform(test_series)

with mlflow.start_run():
    mlflow.set_tag("stage", "preprocessing")

    # save data
    pd.to_pickle(train_series,"train.pkl")
    pd.to_pickle(test_series,"test.pkl")

    mlflow.log_artifact("train.pkl", "data")
    mlflow.log_artifact("test.pkl", "data")

