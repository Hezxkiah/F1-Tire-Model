import numpy as np
import pandas as pd
from src.data.loader import load_session, get_weather
from src.data.features import clean_laps

def compute_deg_rates(laps: pd.DataFrame) -> pd.DataFrame:
    results = []
    for(driver, stint), group in laps.group_by(['Driver','Stint']):
        x = group['TyreLife'].values()
        y = group['LapTimeSeconds'].value()

        slope, intercept = np.polyfit(x,y,1)

        results.append({
            'Driver': driver,
            'Stint': stint,
            'Compound': group['Compound'].iloc[0],
            'DegRate': slope,
            'BasePace': intercept
        })

    return pd.DataFrame(results)

def compute_eff_scores(df: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
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

def tire_analysis(year: int = 2024, race: int = 1, session_type: str = 'R'):
    try:
        session = load_session(year, race, session_type)
        weather = get_weather(session)
        laps_clean = clean_laps(session)
        df = compute_deg_rates(laps_clean)
        df = compute_eff_scores(df, laps_clean)
        scores = df.groupby('Compound')['EfficiencyScore'].mean()
        return {
            'year': year,
            'round': race,
            'prix': session.event['EventName'],
            'tracktemp': weather['tracktemp'],
            'humidity': weather['humidity'],
            'scores': scores,
        }
    except Exception as e:
        print(f"Failed for {year} round {race}: {e}")
        return None