## Requirements

- Python 3.8+
- pandas
- numpy
- tensorflow        # includes Keras API
- scikit-learn      # for train_test_split, StandardScaler, metrics
- matplotlib
- seaborn
- joblib            # for saving/loading scalers



Accurate short-range runoff forecasts are essential for flood warning and water-resources management. The NOAA National Water Model (NWM) provides 1–18 h lead-time streamflow forecasts, but these often exhibit systematic underprediction and diminished variance at longer leads. In this project, I developed a deep-learning post-processor that learns the residual errors between NWM forecasts and USGS observations to produce corrected runoff forecasts with reduced bias and improved skill.
