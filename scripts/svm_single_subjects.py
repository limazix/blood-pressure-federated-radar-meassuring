#%%
import os

import pandas as pd
import numpy as np

import joblib
from dask.distributed import Client

import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.svm import SVR

import scipy.io as sio
from scipy import signal

import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# %%
DATA_ROOT_PATH=os.path.abspath('./data')

SUBJECT_ID_PREFIX='GDN00'
SUBJECT_IDS=[SUBJECT_ID_PREFIX + str(i) if i > 9 else '{}0{}'.format(SUBJECT_ID_PREFIX, i) for i in range(1, 31)]

SCENARIOS=['Resting', 'Valsalva', 'Apnea', 'TiltUp', 'TiltDown']

SUBJECT_ID=SUBJECT_IDS[3]
SCENARIO=SCENARIOS[0]

SCENARIO_PATH=os.path.join(DATA_ROOT_PATH, SUBJECT_ID, '{}_1_{}.mat'.format(SUBJECT_ID, SCENARIO))

# %%
def get_all_files():
    paths = []
    for dir in os.listdir(DATA_ROOT_PATH):
        filepath = os.path.join(DATA_ROOT_PATH, dir)
        if os.path.isdir(filepath):
            for filename in os.listdir(filepath):
                if filename.endswith('.mat'):
                    paths.append(os.path.join(filepath, filename))
    return paths

def load_data(filepath):
    return sio.loadmat(filepath)

def mat_to_dict(mat):
    return {k: np.array(v).flatten() for k, v in mat.items() if k[0] != '_'}

def resample(data, sample_rate):
    return signal.resample(data, sample_rate)

def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

class PulseTransformer(BaseEstimator, TransformerMixin):
    "class used to compute the pulse from Q and I"
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['pulse'] = np.arctan(X_['radar_q']/X_['radar_i'])
        return X_

class FrameStatsTransformer(BaseEstimator, TransformerMixin):
    "class used to compute the frame data stats: mean and std"
    def __init__(self, frame_size) -> None:
        super().__init__()
        self.frame_size = frame_size

    def fit(self, X, y=None):
        return self

    def column_stats(self, data):
        means = []
        stds = []
        for index, item in enumerate(data):
            frame = None
            if index < self.frame_size:
                frame = data[:index-1]
            else:
                frame = data[:index-self.frame_size]
            means.append(frame.mean())
            stds.append(frame.std())
        return means, stds

    def transform(self, X:pd.DataFrame, y=None):
        X_ = X.copy()
        
        for column in X_.columns:
            means, stds = self.column_stats(X_[column])
            X_['mean_{}'.format(column)] = means
            X_['std_{}'.format(column)] = stds

        return X_.fillna(0)

# %%
data = mat_to_dict(load_data(SCENARIO_PATH))
num_samples = len(data['tfm_bp'])
df_data = pd.DataFrame({
    'radar_i': resample(data['radar_i'], num_samples),
    'radar_q': resample(data['radar_q'], num_samples),
    'bp': data['tfm_bp']
})

# %%

pipe = Pipeline(steps=[
    ('PulseTransformer', PulseTransformer()),
    ('FrameStatsTransform', FrameStatsTransformer(frame_size=100)),
    ('SVR', SVR(gamma='auto'))
], verbose=True)

# %%
train_size = int(np.round(df_data.shape[0] * 0.8))

X_train = df_data[:train_size].drop('bp', axis=1)
y_train = df_data['bp'][:train_size]

X_test = df_data[train_size:].drop('bp', axis=1)
y_test = df_data['bp'][train_size:]

# %%
pipe.fit(X_train, y_train)

# %%
y_pred = pipe.predict(X_test)
regression_results(y_test, y_pred)
# %%
