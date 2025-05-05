import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# custom NSE loss function for model
def nse_loss(y_true, y_pred):
    num   = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    denom = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)), axis=-1)
    return num / (denom + 1e-6) #make sure no divide by zero

# How long the sequence is for LSTM training
SEQUENCE_LENGTH = 6  

# load in the model
df = pd.read_csv('new_test_data_57.csv')
model = tf.keras.models.load_model(
    'postprocessing_residual_lstm_model_57.keras',
    custom_objects={'nse_loss': nse_loss}
)
# Load scalers
seq_scaler    = joblib.load('seq_scaler.pkl')
static_scaler = joblib.load('static_scaler.pkl')
y_scaler      = joblib.load('y_scaler_resid_lstm.pkl')

# clean data (for safety maybe not neccesary)
df = df.replace([np.inf, -np.inf], pd.NA).dropna()
df['lead_time_hr'] = df['lead_time_hr'].round().astype(int)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['init_time'] = df['DateTime'] - pd.to_timedelta(df['lead_time_hr'], unit='h')
df['streamflow_value'] = df['streamflow_value'].clip(lower=0.01)


# lookup observed USGS at initialization
usgs_lookup = df[['DateTime','USGSFlowValue']].drop_duplicates()
usgs_lookup.columns = ['init_time','init_usgs']
# merge initialization USGS into main df
(df) = df.merge(usgs_lookup, on='init_time', how='left').dropna(subset=['init_usgs'])

# Precipitation was too much of a factor, so i scaled it down
df['prcp_mm_per_hr'] = df['prcp_mm_day'] / 24.0
df['prcp_sqrt']       = np.sqrt(df['prcp_mm_per_hr'].clip(lower=0))


# Making the data series
usgs_series = df.drop_duplicates(subset='DateTime').set_index('DateTime')['USGSFlowValue']
seq_list = []
static_list = []
index_list = []
for idx, row in df.iterrows():
    init = row['init_time']
    times = [init - pd.Timedelta(hours=h) for h in reversed(range(SEQUENCE_LENGTH))]
    vals = usgs_series.reindex(times).values
    if np.isnan(vals).any():
        continue
    seq_list.append(vals)
    static_list.append([
        row['streamflow_value'],
        row['lead_time_hr'],
        row['init_usgs'],
        row['prcp_sqrt']
    ])
    index_list.append(idx)

X_seq    = np.array(seq_list).reshape(-1, SEQUENCE_LENGTH, 1)
X_static = np.array(static_list)

# making scalers
X_seq_flat      = X_seq.reshape(-1,1)
X_seq_scaled    = seq_scaler.transform(X_seq_flat).reshape(X_seq.shape)
X_static_scaled = static_scaler.transform(X_static)

# Run the model
y_pred_scaled = model.predict({'usgs_sequence': X_seq_scaled, 'static_features': X_static_scaled})
y_pred        = y_scaler.inverse_transform(y_pred_scaled).flatten()

df_valid = df.loc[index_list].copy()
df_valid['predicted_residual'] = y_pred
df_valid['corrected'] = df_valid['streamflow_value'] + df_valid['predicted_residual']

# Make all plots show in slide
# Some of these I altered to not show outliers so it would fit nicer
df_melt = df_valid.melt(
    id_vars='lead_time_hr',
    value_vars=['USGSFlowValue','streamflow_value','corrected'],
    var_name='Source', value_name='Runoff'
)
plt.figure(figsize=(15,6))
sns.boxplot(data=df_melt, x='lead_time_hr', y='Runoff', hue='Source', showfliers=False)
plt.title('Runoff Distribution by Lead Time')
plt.xlabel('Lead Time (hrs)')
plt.ylabel('Runoff (cfs)')
plt.legend(title='Source')
plt.tight_layout()
plt.show()

# Get the data for the box plots
df_valid['date'] = df_valid['DateTime'].dt.date
metrics_rows = []
for (lead, date), group in df_valid.groupby(['lead_time_hr','date']):
    obs  = group['USGSFlowValue'].values
    nwm  = group['streamflow_value'].values
    corr = group['corrected'].values
    for label, series in [('NWM',nwm), ('Corrected',corr)]:
        metrics_rows.append({
            'lead_time_hr': lead,
            'date': date,
            'Type': label,
            'RMSE':  np.sqrt(np.mean((series-obs)**2)),
            'PBIAS': 100 * np.sum(series-obs) / np.sum(obs),
            'NSE':   1 - np.sum((series-obs)**2) / np.sum((obs-np.mean(obs))**2),
            'CC':    np.corrcoef(obs,series)[0,1]
        })
metrics_df = pd.DataFrame(metrics_rows)

# plot that data
for metric in ['RMSE','PBIAS','CC']:
    plt.figure(figsize=(12,6))
    sns.boxplot(
        data=metrics_df,
        x='lead_time_hr', y=metric, hue='Type',
        # whis=[5,95],      # whiskers at 5th and 95th percentiles
        showfliers=False  # hide outlier points
    )
    plt.title(f'{metric} Distribution by Lead Time (Daily, 5–95%)')
    plt.xlabel('Lead Time (hrs)')
    plt.ylabel(metric)
    plt.legend(title='Forecast')
    plt.tight_layout()
    plt.show()

# The NSE looked horrible when it was individual values
# So I sectioned it per month
df_valid['month'] = df_valid['DateTime'].dt.to_period('M').astype(str)
nse_rows = []
for (lead, mon), group in df_valid.groupby(['lead_time_hr','month']):
    obs = group['USGSFlowValue'].values
    nwm = group['streamflow_value'].values
    corr = group['corrected'].values
    if len(obs) < 2:
        continue
    nse_val_nwm  = 1 - np.sum((nwm-obs)**2)  / np.sum((obs-np.mean(obs))**2)
    nse_val_corr = 1 - np.sum((corr-obs)**2) / np.sum((obs-np.mean(obs))**2)
    nse_rows.append({'lead_time_hr':lead, 'period':mon, 'Type':'NWM',      'NSE':nse_val_nwm})
    nse_rows.append({'lead_time_hr':lead, 'period':mon, 'Type':'Corrected','NSE':nse_val_corr})
nse_df = pd.DataFrame(nse_rows)

plt.figure(figsize=(12,6))
sns.boxplot(
    data=nse_df,
    x='lead_time_hr', y='NSE', hue='Type',
    whis=[5,95],      # whiskers at 5th and 95th percentiles
    showfliers=False  # hide outlier points
)
plt.title('Monthly NSE Distribution by Lead Time (5–95%)')
plt.xlabel('Lead Time (hrs)')
plt.ylabel('Nash-Sutcliffe Efficiency')
plt.axhline(0, color='grey', linestyle='--')
plt.ylim(-1,1)
plt.legend(title='Forecast')
plt.tight_layout()
plt.show()

print("✅ Evaluation complete: RMSE/PBIAS/CC daily; NSE monthly.")
