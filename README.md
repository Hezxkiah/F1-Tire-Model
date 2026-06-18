# F1 Tire Efficiency Analysis

A data analysis project that uses the [FastF1](https://docs.fastf1.dev/) library to quantify tire compound degradation rates across Formula 1 races, using linear regression to determine which tire compound performs most efficiently at a given circuit.

---

## Project Goal

F1 tire strategy is one of the most consequential decisions a team makes during a race weekend. The goal of this project is to move beyond intuition and quantify tire efficiency using real lap data, specifically, how many seconds per lap each compound loses as it ages.

The core question: which tire compound degrades the least, and therefore offers the best strategic value at a given circuit?

---

## Methodology

### 1. Data Collection
Race session data is loaded via FastF1, which pulls official F1 timing and telemetry data. The following fields are extracted per lap:

| Field | Description |
|---|---|
| `Driver` | Three-letter driver code |
| `LapNumber` | Lap number within the race |
| `Stint` | Which stint the lap belongs to |
| `LapTime` | Raw lap duration (timedelta) |
| `Compound` | Tire compound used (SOFT, MEDIUM, HARD, etc.) |
| `TyreLife` | How many laps that set of tires has been on the car |

### 2. Data Cleaning
Several types of laps are removed before analysis because they distort true race pace:
- Laps with missing lap time or compound data (NaN rows)
- Laps with zero or negative lap times
- Laps in the top 5% slowest (safety car periods, pit entry/exit laps, incidents)

### 3. Degradation Rate Calculation

Tire degradation is modeled as a linear relationship between tire age and lap time:

$$\text{LapTime} = \beta_0 + \beta_1 \times \text{TyreLife}$$

Where:
- $\beta_1$ (the slope) = **degradation rate** in seconds per lap
- $\beta_0$ (the intercept) = **base pace** — the theoretical lap time on a fresh tire

A higher slope means faster degradation. A negative slope (pace improving with age) typically indicates a data artifact, such as mixed driver speeds across stints.

#### Why Per-Stint Regression?

A naive approach fits one regression line across all drivers on a given compound. This is flawed because different drivers have different raw pace. Mixing VER's 94s laps with a backmarker's 97s laps makes the compound appear to "change pace" when it's really just different drivers.

The correct approach groups data by Driver + Stint, fits a regression within each individual stint, then averages the slopes by compound. This isolates the tire aging effect from driver speed differences.

### 4. Results

For the 2021 Bahrain Grand Prix:

| Compound | Avg Degradation Rate | Interpretation |
|---|---|---|
| HARD | 0.0659 s/lap | Loses ~1.32s over 20 laps |
| SOFT | 0.1161 s/lap | Loses ~2.32s over 20 laps |

### Conclusion

By utilizing linear regression on lap data collected via FastF1, I concluded that in the 2024 Bahrain Grand Prix, HARD tires completed an average-length stint approximately 21 seconds faster in aggregate than SOFT tires. Despite SOFT tires offering a faster base pace, their degradation rate of 0.116 s/lap versus 0.066 s/lap for HARD meant that over a full stint, the HARD compound's durability outweighed the SOFT's early pace advantage, making it the more strategically efficient choice at this circuit.

---

## Project Structure

```
F1 Model/
│
├── model.ipynb          # Main analysis notebook
├── README.md            # This file
└── venv/                # Python virtual environment (not tracked)
```
## Dependencies

| Library | Purpose |
|---|---|
| `fastf1` | Load F1 timing, lap, and telemetry data |
| `pandas` | Data manipulation and groupby analysis |
| `numpy` | Linear regression via `np.polyfit` |
| `matplotlib` | Visualization (in progress) |

---

## Key Concepts

**Degradation Rate** — The slope of a linear regression fit between TyreLife (x) and LapTimeSeconds (y) within a single stint. Measured in seconds per lap.

**Base Pace** — The intercept of the same regression. Represents the theoretical lap time on a brand new tire for that driver in that stint.

**Stint** — A continuous sequence of laps on the same set of tires between pit stops.

**Efficiency** — A compound is considered more efficient if it has a lower degradation rate, meaning it loses less time per lap as it ages, even if its base pace is slightly slower.

---

## Planned Extensions

- [ ] Combine degradation rate and base pace into a single efficiency score per compound
- [ ] Scale analysis across multiple races in a season to build circuit-by-circuit tire profiles
- [ ] Add weather data integration (track temperature vs degradation rate)
- [ ] Visualize degradation curves per compound with scatter plots and regression lines

---

## Data Source

Session data is sourced via [FastF1](https://docs.fastf1.dev/), an open-source Python library that interfaces with the official F1 timing feed. Data is intended for educational and personal analysis purposes.

---

## Author

Hezekiah Gitenyi — Computer Science & Math student at Texas A&M University
