import pandas as pd
from src.models.linear import tire_analysis

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
    return season_df, best_compound

if __name__ == '__main__':
    season_df, best_compound = run_season(2024)
    print(season_df)
    print(best_compound)