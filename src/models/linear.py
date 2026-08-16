import numpy as np
import pandas as pd
from src.data.loader import load_session, get_weather
from src.data.features import clean_laps
from src.data.features import eff_scores

def compute_deg_rates(laps: pd.DataFrame) -> pd.DataFrame:
    results = []
    for(driver, stint), group in laps.groupby(['Driver','Stint']):
        x = group['TyreLife'].values
        y = group['LapTimeSeconds'].values

        slope, intercept = np.polyfit(x,y,1)

        results.append({
            'Driver': driver,
            'Stint': stint,
            'Compound': group['Compound'].iloc[0],
            'DegRate': slope,
            'BasePace': intercept
        })

    return pd.DataFrame(results)

def tire_analysis(year: int = 2024, race: int = 1, session_type: str = 'R'):
    try:
        session = load_session(year, race, session_type)
        weather = get_weather(session)
        laps_clean = clean_laps(session)
        df = compute_deg_rates(laps_clean)
        df = eff_scores(df, laps_clean)
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

def run_season(year: int = 2024):
    season_results = []
    for race in range(1, 25):
        result = tire_analysis(year, race)
        if result is None:
            continue
        for compound, score in result['scores'].items():
            season_results.append({
                'year': result['year'],
                'round': result['round'],
                'prix': result['prix'],
                'tracktemp': result['tracktemp'],
                'humidity': result['humidity'],
                'compound': compound,
                'score': score,
            })
    season_df = pd.DataFrame(season_results)
    season_df = season_df[~season_df['compound'].isin(['INTERMEDIATE', 'WET'])]
    best_compound = season_df.loc[season_df.groupby('round')['score'].idxmin()]
    compound_avg = season_df.groupby('compound')['score'].mean().sort_values()
    return season_df, best_compound,compound_avg

if __name__ == '__main__':
    season_df, best_compound, compound_avg = run_season(2024)
    print(season_df)
    print(best_compound)
    print(compound_avg)