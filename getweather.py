import pandas as pd

df = pd.read_csv('test_data_41.csv', parse_dates=['DateTime'])

df_pr = pd.read_csv('rain41.csv')  


df_pr['date'] = pd.to_datetime(df_pr['year']*1000 + df_pr['yday'], format='%Y%j').dt.date
df_pr.rename(columns={'prcp (mm/day)': 'prcp_mm_day'}, inplace=True)

df['date'] = df['DateTime'].dt.date

df_merged = df.merge(
    df_pr[['date','prcp_mm_day']],
    on='date',
    how='left'
)

df_merged.drop(columns=['date'], inplace=True)

df_merged.to_csv('new_test_data_41.csv', index=False)
