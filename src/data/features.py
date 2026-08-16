import pandas as pd

def clean_laps(session) -> pd.DataFrame:
    cols = ['Driver', 'LapNumber', 'Stint', 'LapTime', 'Compound', 'TyreLife','LapStartTime']
    laps = session.laps[cols].copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps = laps.dropna(subset=['LapTimeSeconds','Compound', 'LapStartTime'])
    laps = laps[laps['LapTimeSeconds'] > 0]
    threshold = laps['LapTimeSeconds'].quantile(0.95)
    laps = laps[laps['LapTimeSeconds'] < threshold]

    weather = session.weather_data[['Time', 'TrackTemp', 'AirTemp', 'Humidity', 'Rainfall']].copy()
    laps = laps.sort_values('LapStartTime')
    weather = weather.sort_values('Time')

    laps = pd.merge_asof(
        laps, weather,
        left_on='LapStartTime', right_on='Time',
        direction='nearest'
    )
    laps = laps.drop(columns=['Time'])
    laps = laps[~laps['Compound'].isin(['INTERMEDIATE', 'WET'])]

    return laps

def eff_scores(df: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    stint_lengths = (
        laps.groupby(['Driver','Stint']).size().reset_index(name='StintLength')
    )
    df = df.merge(stint_lengths, on=['Driver', 'Stint'])
    avg_stint = df['StintLength'].mean()

    df['EfficiencyScore'] = df.apply(
        lambda row: (row['BasePace'] * avg_stint) + 0.5 * (row['DegRate'] * avg_stint ** 2),
        axis = 1
    )

    return df
