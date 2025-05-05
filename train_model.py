import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping  || took out after seeing where it stopped
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

"""
IMPORTANT NOTE

I trained the models for 57 and 41 separately, since they have 
different patterns (presumably) and different precipitation
You can simply switch the data you load in to test each model
The current github has the model for 57. 
"""


# How long the sequence is for LSTM training
SEQUENCE_LENGTH = 6  

# load data (is probably already clean of NaN but just making sure)
df = pd.read_csv('new_train_data_57.csv')
df = df.replace([np.inf, -np.inf], pd.NA).dropna()

# Fix Datetime stuff
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['init_time'] = df['DateTime'] - pd.to_timedelta(df['lead_time_hr'], unit='h')
# build lookup for initialization USGS
usgs_lookup = df[['DateTime','USGSFlowValue']].drop_duplicates()
usgs_lookup.columns = ['init_time','init_usgs']
df = df.merge(usgs_lookup, on='init_time', how='left').dropna(subset=['init_usgs'])

# precipitation made model much worse, so i lowered the numbers
df['prcp_mm_per_hr'] = df['prcp_mm_day'] / 24.0
df['prcp_sqrt'] = np.sqrt(df['prcp_mm_per_hr'].clip(lower=0))


# create series of unique USGS observations for sequence lookup
usgs_series = df.drop_duplicates(subset='DateTime').set_index('DateTime')['USGSFlowValue']

seq_list = []
static_list = []
y_list = []
for idx, row in df.iterrows():
    init = row['init_time']
    # gather past SEQUENCE_LENGTH USGS values
    times = [init - pd.Timedelta(hours=h) for h in reversed(range(SEQUENCE_LENGTH))]
    vals = usgs_series.reindex(times).values
    if np.isnan(vals).any(): #again, just for safety may not do anything
        continue
    seq_list.append(vals)
    # static features
    static_list.append([
        row['streamflow_value'],
        row['lead_time_hr'],
        row['init_usgs'],
        row['prcp_sqrt']
    ])
    # target residual
    y_list.append(row['USGSFlowValue'] - row['streamflow_value'])

# convert to arrays
X_seq = np.array(seq_list)[:, :, np.newaxis]  # shape (N, T, 1)
X_static = np.array(static_list)               # shape (N, 4)
y = np.array(y_list)                           # shape (N,)

# scale sequence values
seq_scaler = StandardScaler()
X_seq_flat = X_seq.reshape(-1,1)
X_seq_scaled = seq_scaler.fit_transform(X_seq_flat).reshape(X_seq.shape)

# scale static features
static_scaler = StandardScaler()
X_static_scaled = static_scaler.fit_transform(X_static)

# scale target residual
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1,1)).flatten()

# save scalers
joblib.dump(seq_scaler, 'seq_scaler.pkl')
joblib.dump(static_scaler, 'static_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler_resid_lstm.pkl')

# Split into tarining and validation (80 20 split)
X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = \
    train_test_split(X_seq_scaled, X_static_scaled, y_scaled, test_size=0.2, random_state=42)

# shape the input
seq_input = tf.keras.Input(shape=(SEQUENCE_LENGTH,1), name='usgs_sequence')
static_input = tf.keras.Input(shape=(4,), name='static_features')

# LSTM branch
x1 = tf.keras.layers.LSTM(64)(seq_input)
# static branch
x2 = tf.keras.layers.Dense(32, activation='relu')(static_input)

# combine branches
x = tf.keras.layers.Concatenate()([x1, x2])
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
out = tf.keras.layers.Dense(1, name='predicted_residual')(x)

model = tf.keras.Model(inputs=[seq_input, static_input], outputs=out)

#custom loss function
def nse_loss(y_true, y_pred):
    residual = tf.square(y_true - y_pred)
    denom    = tf.square(y_true - tf.reduce_mean(y_true))
    return tf.reduce_sum(residual) / (tf.reduce_sum(denom) + 1e-6)

model.compile(
    optimizer='adam',
    loss=nse_loss,
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)




# train
history = model.fit(
    {'usgs_sequence': X_seq_train, 'static_features': X_static_train},
    y_train,
    validation_data=({'usgs_sequence': X_seq_val, 'static_features': X_static_val}, y_val),
    epochs=100,
    batch_size=64,
    verbose=1
)

# Plotting 
plt.figure(figsize=(10,6))
plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE (std residual)')
plt.title('LSTM Residual Model Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# save it
model.save('postprocessing_residual_lstm_model_57.keras')
