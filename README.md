# F1 Tire Efficiency Analysis

This is a data analysis project that uses the [FastF1](https://docs.fastf1.dev/) library to quantify tire compound degradation rates across Formula 1 races, using linear regression to determine which tire compound performs most efficiently at a given circuit.

---

## Project Goal

F1 tire strategy is one of the most influential decisions a team makes during a race weekend. The goal of this project is to quantify tire efficiency using real lap data, specifically, how many seconds per lap each compound loses as it ages.

The core question: which tire compound degrades the least, and therefore offers the best strategic value at a given circuit?

---

## Methodology

### 1. Data Collection
Race session data is loaded via FastF1 API, which pulls official F1 timing and telemetry data. The following fields are extracted per lap:

| Field | Description |
|---|---|
| `Driver` | Three-letter driver code |
| `LapNumber` | Lap number within the race |
| `Stint` | Which stint the lap belongs to |
| `LapTime` | Raw lap duration (timedelta) |
| `Compound` | Tire compound used (SOFT, MEDIUM, HARD, etc.) |
| `TireLife` | How many laps that set of tires has been on the car |

### 2. Data Cleaning
Several types of laps are removed before analysis because they distort true race pace:
- Laps with missing lap time or compound data (NaN rows)
- Laps with zero or negative lap times
- Laps in the top 5% slowest (safety car periods, pit entry/exit laps, incidents)

### 3. Linear Degradation Rate Calculation

Tire degradation is modeled as a linear relationship between tire age and lap time:

$$\text{LapTime} = \beta_0 + \beta_1 \times \text{TireLife}$$

Where:
- $\beta_1$ (the slope) = **degradation rate** in seconds per lap
- $\beta_0$ (the intercept) = **base pace** — the theoretical lap time on a fresh tire

A higher slope means faster degradation while a negative slope pace means improvment with age.

## Dependencies

| Library | Purpose |
|---|---|
| `fastf1` | Load F1 timing, lap, and telemetry data |
| `pandas` | Data manipulation and groupby analysis |
| `numpy` | Linear regression via `np.polyfit` |
| `matplotlib` | Visualization (in progress) |

---

## Data Source

Session data is sourced via [FastF1](https://docs.fastf1.dev/), an open-source Python library that interfaces with the official F1 timing feed. Data is intended for educational and personal analysis purposes.

---

## Author

Hezekiah Gitenyi — Computer Science student at Texas A&M University
