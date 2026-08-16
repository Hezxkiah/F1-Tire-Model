import numpy as np
import pandas as pd
import fastf1 as f1
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from src.data.loader import load_session
from src.data.features import clean_laps

FEATURE_COLS = ['TyreLife', 'LapNumber', 'TrackTemp', 'AirTemp',
                'Compound_HARD', 'Compound_MEDIUM', 'Compound_SOFT']
MODEL_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    random_state=42,
    eval_metric="mae"
)


def build_dataset(year: int) -> pd.DataFrame:
    """Load and clean every race in a season into one dataframe."""
    schedule = f1.get_event_schedule(year)
    all_laps = []

    for rnd in schedule['RoundNumber'].to_list():
        try:
            session = load_session(year, rnd)
            laps = clean_laps(session)
            laps['RaceId'] = rnd
            all_laps.append(laps)
        except Exception as e:
            print(f"Round {rnd} failed: {e}")

    laps_full = pd.concat(all_laps, ignore_index=True)

    compound_dummies = pd.get_dummies(laps_full['Compound'], prefix='Compound')
    for col in ['Compound_HARD', 'Compound_MEDIUM', 'Compound_SOFT']:
        if col not in compound_dummies.columns:
            compound_dummies[col] = False
    laps_full = pd.concat([laps_full, compound_dummies], axis=1)

    laps_full['RaceMedianLapTime'] = laps_full.groupby('RaceId')['LapTimeSeconds'].transform('median')
    laps_full['LapTimeDelta'] = laps_full['LapTimeSeconds'] - laps_full['RaceMedianLapTime']

    return laps_full


def train_model(X_train, y_train, eval_set) -> XGBRegressor:
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def run_cross_validation(laps_full: pd.DataFrame, n_splits: int = 5):
    X = laps_full[FEATURE_COLS]
    y = laps_full['LapTimeDelta']
    groups = laps_full['RaceId']

    gkf = GroupKFold(n_splits=n_splits)
    fold_maes = []

    for fold_num, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = train_model(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        fold_maes.append(mae)

        test_races = laps_full.iloc[test_idx]['RaceId'].unique()
        print(f"Fold {fold_num}: MAE={mae:.3f}, best_iter={model.best_iteration}, races={test_races}")

    print(f"\nAverage MAE: {np.mean(fold_maes):.3f}  |  Std: {np.std(fold_maes):.3f}")
    return fold_maes


def simulate_strategy(model, strategy, track_temp, air_temp, race_median, pit_loss=22.0):
    """Predict total race time for a candidate strategy."""
    total_time = 0
    lap_number = 0

    for compound, stint_length in strategy:
        for tyre_life in range(1, stint_length + 1):
            lap_number += 1
            row = {
                'TyreLife': tyre_life,
                'LapNumber': lap_number,
                'TrackTemp': track_temp,
                'AirTemp': air_temp,
                'Compound_HARD': compound == 'HARD',
                'Compound_MEDIUM': compound == 'MEDIUM',
                'Compound_SOFT': compound == 'SOFT',
            }
            X_row = pd.DataFrame([row])[FEATURE_COLS]
            predicted_delta = model.predict(X_row)[0]
            total_time += race_median + predicted_delta
        total_time += pit_loss

    total_time -= pit_loss  # no pit stop needed after the final stint
    return total_time


def get_actual_strategy(actual_race_data: pd.DataFrame):
    """Pick the most-complete driver in a race and return their real stint pattern."""
    lap_counts = actual_race_data.groupby('Driver')['LapNumber'].count().sort_values(ascending=False)
    driver = lap_counts.index[0]
    driver_data = actual_race_data[actual_race_data['Driver'] == driver].sort_values('LapNumber')

    stints = driver_data.groupby('Stint').agg(Compound=('Compound', 'first'), Laps=('LapNumber', 'count'))
    pattern = [(row.Compound, int(row.Laps)) for row in stints.itertuples()]

    return {
        'driver': driver,
        'race_length': int(driver_data['LapNumber'].max()),
        'actual_total_time': driver_data['LapTimeSeconds'].sum(),
        'pattern': pattern
    }


def backtest_race(laps_full: pd.DataFrame, test_race: int) -> dict:
    """Train on every race except test_race, then compare predicted vs actual for that race."""
    train_data = laps_full[laps_full['RaceId'] != test_race]
    actual_race_data = laps_full[laps_full['RaceId'] == test_race]

    if actual_race_data.empty:
        return None

    X_train = train_data[FEATURE_COLS]
    y_train = train_data['LapTimeDelta']
    model = train_model(X_train, y_train, eval_set=[(X_train, y_train)])

    race_median = actual_race_data['LapTimeSeconds'].median()
    track_temp = actual_race_data['TrackTemp'].mean()
    air_temp = actual_race_data['AirTemp'].mean()

    actual = get_actual_strategy(actual_race_data)
    predicted_time = simulate_strategy(model, actual['pattern'], track_temp, air_temp, race_median)

    diff = predicted_time - actual['actual_total_time']
    pct_diff = (diff / actual['actual_total_time']) * 100

    return {
        'race': test_race,
        'driver': actual['driver'],
        'actual_time': actual['actual_total_time'],
        'predicted_time': predicted_time,
        'diff': diff,
        'pct_diff': pct_diff
    }


def run_backtests(laps_full: pd.DataFrame, test_races: list) -> pd.DataFrame:
    results = []
    for race in test_races:
        result = backtest_race(laps_full, race)
        if result is None:
            print(f"Race {race}: no data, skipping")
            continue
        results.append(result)
        print(f"Race {result['race']} ({result['driver']}): "
              f"actual={result['actual_time']:.1f}s, predicted={result['predicted_time']:.1f}s, "
              f"diff={result['diff']:+.1f}s ({result['pct_diff']:+.2f}%)")

    results_df = pd.DataFrame(results)
    print("\n--- Summary ---")
    print(results_df[['race', 'driver', 'diff', 'pct_diff']])
    print(f"\nMean absolute diff: {results_df['diff'].abs().mean():.1f}s")
    print(f"Mean pct diff: {results_df['pct_diff'].mean():+.2f}%")
    return results_df


if __name__ == "__main__":
    laps_full = build_dataset(year=2024)

    print("=== Cross-validation ===")
    run_cross_validation(laps_full)

    print("\n=== Backtesting ===")
    test_races = [2, 10, 12, 19, 20, 7, 8, 9, 17, 24]
    run_backtests(laps_full, test_races)