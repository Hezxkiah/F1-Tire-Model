import pandas as pd

def clean_laps(session) -> pd.DataFrame:
    cols = ['Driver', 'LapNumber', 'Stint', 'LapTime', 'Compound', 'TyreLife',]
    laps = session.laps[cols].copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps = laps.dropna(subset=['LapTimeSeconds','Compound'])
    laps = laps[laps['LapTimeSeconds'] > 0]
    threshold = laps['LapTimeSeconds'].quantile(0.95)
    laps = laps[laps['LapTimeSeconds'] < threshold]
