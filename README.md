[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://health-disparity-analysis-2025.streamlit.app/)


# Health Disparity Analysis

Quantitative analysis of health disparities across demographic and geographic dimensions in the United States.

## Project Overview

This project analyzes health inequities across three key dimensions:
- Geographic disparities (metro vs. nonmetro)
- Racial and ethnic disparities
- Income-based disparities

The analysis uses data from America's Health Rankings 2025 Annual Report to calculate disparity metrics and create a composite index ranking states by overall health inequity.

## Features

- Composite disparity index combining geographic, racial, and income dimensions
- Interactive visualizations of state-level disparities
- State-specific deep-dive analysis
- Identification of health measures with largest gaps across demographic groups

## Data Source

America's Health Rankings 2025 Annual Report
- 82,054 records
- 1,578 unique health measures
- 51 states plus DC
- Demographic breakdowns across race, income, geography, age, education, and other factors

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Analysis

Generate disparity metrics:
```bash
python disparity_analysis.py
```

This creates four CSV files:
- `metro_nonmetro_gaps.csv`: Geographic disparities
- `racial_disparities.csv`: Racial health gaps
- `income_disparities.csv`: Income-based disparities
- `composite_disparity_index.csv`: Combined disparity rankings

### Launch Dashboard

Run the interactive Streamlit application:
```bash
streamlit run streamlit_app.py
```

## Methodology

### Disparity Calculation

For each dimension (geographic, racial, income), the analysis:
1. Identifies relevant demographic splits in the data
2. Calculates gaps between best and worst outcomes within each state
3. Aggregates gaps across multiple health measures
4. Standardizes metrics for comparison

### Composite Index

The composite disparity index:
1. Standardizes each disparity dimension (mean=0, sd=1)
2. Averages standardized scores across dimensions
3. Ranks states from highest to lowest disparity

## Key Findings

Top states by composite disparity:
1. South Dakota
2. Tennessee
3. Kentucky
4. Alabama
5. Arkansas

## Technical Stack

- Python 3.12
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Standardization and statistical methods
- Streamlit: Interactive dashboard
- Plotly: Visualizations

## Project Structure

```
health_disparity_analysis/
├── disparity_analysis.py      # Core analysis functions
├── streamlit_app.py            # Interactive dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── [generated CSV files]       # Analysis outputs
```

## Future Enhancements

Potential extensions:
- Time-series analysis of disparity trends
- Machine learning models to predict disparity patterns
- Additional demographic dimensions (disability, education, age)
- County-level analysis
- Policy intervention impact assessment

## Author

Jeffrey Olney
GitHub | LinkedIn

## License

This project is available for educational and research purposes.
