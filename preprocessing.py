import pandas as pd
import glob
import os

station_folder = '.././21609641' #'./20380357'  # './21609641'

# USGS data
usgs_file = os.path.join(station_folder, '11266500_Strt_2021-04-20_EndAt_2023-04-21.csv')
usgs = pd.read_csv(usgs_file, parse_dates=['DateTime'])
# Make it hourly
usgs = usgs[usgs['DateTime'].dt.minute == 0]
# Remove timezone if present
usgs['DateTime'] = usgs['DateTime'].dt.tz_localize(None)
# Unit Conversion (only for 57)
#usgs['USGSFlowValue'] = usgs['USGSFlowValue'] * 35.3147

# NWM Data
nwm_files = glob.glob(os.path.join(station_folder, 'streamflow_*.csv'))

all_merged = []

for file in nwm_files:
    nwm = pd.read_csv(file)

    # Parse timestamps
    nwm['model_output_valid_time'] = pd.to_datetime(
        nwm['model_output_valid_time'],
        format='%Y-%m-%d_%H:%M:%S',
        errors='coerce'
    )
    nwm['model_initialization_time'] = pd.to_datetime(
        nwm['model_initialization_time'],
        format='%Y-%m-%d_%H:%M:%S',
        errors='coerce'
    )
    nwm['model_output_valid_time'] = nwm['model_output_valid_time'].dt.tz_localize(None)
    nwm['model_initialization_time'] = nwm['model_initialization_time'].dt.tz_localize(None)

    # Calculate lead time in hours
    nwm['lead_time_hr'] = (
        nwm['model_output_valid_time'] - nwm['model_initialization_time']
    ).dt.total_seconds() / 3600

    # Keep only forecasts with 1–18 hour lead time
    nwm = nwm[nwm['lead_time_hr'].between(1, 18)]

    # Merge with USGS
    merged = pd.merge(
        usgs,
        nwm,
        how='inner',
        left_on='DateTime',
        right_on='model_output_valid_time'
    )

    # Calculate error
    merged['error'] = merged['streamflow_value'] - merged['USGSFlowValue']

    # Final columns
    final = merged[['DateTime', 'streamflow_value', 'USGSFlowValue', 'error', 'lead_time_hr']]

    all_merged.append(final)

# Combine all months
final_df = pd.concat(all_merged, ignore_index=True)

# Train/Test Split
train_end = pd.Timestamp('2022-09-30 23:59:59')
test_start = pd.Timestamp('2022-10-01 00:00:00')

train_df = final_df[final_df['DateTime'] <= train_end]
test_df = final_df[final_df['DateTime'] >= test_start]

# Save
train_df.to_csv('train_data_41.csv', index=False)
test_df.to_csv('test_data_41.csv', index=False)

print("done")
