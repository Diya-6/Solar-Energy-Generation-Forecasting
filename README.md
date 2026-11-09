# Solar-Energy-Generation-Forecasting

### Using real data
CSV with hourly rows and columns: `timestamp, gen_mw, cloud, temp, humidity, wind_speed, wind_dir`
(`cloud` is 0-1, `wind_dir` in degrees). Weather columns must be the forecast that was available for that hour.
Capacity is taken as the maximum observed output.

## Method
1. **Features** (per horizon h, forecast issued at t for target t+h): forecast weather at t+h, solar zenith and
   clear-sky irradiance from latitude, longitude and time, hour and day-of-year, plant output now, mean of last
   3 hours, same hour yesterday, and clear-sky index.
2. **Models**: one `HistGradientBoostingRegressor` per horizon and quantile (10th, 50th, 90th). The median is the point forecast.
3. **Uncertainty**: conformalized quantile regression. The gap between the 10th/90th quantile band and the
   observed values on a held-out calibration block is used to widen or narrow the band to the 80% target.
4. **Evaluation**: chronological split (65% train, 15% calibration, 20% test). Daytime hours only. Baselines are
   smart persistence (last clear-sky index times clear-sky output) and same hour yesterday.

## Results (synthetic data, test set)
| Horizon | nMAE model | nMAE smart persistence | 80% band coverage | Band width (MW) |
|---|---|---|---|---|
| 1 h | 2.4% | 3.4% | 79.4% | 0.75 |
| 3 h | 3.0% | 5.6% | 82.7% | 0.92 |
| 6 h | 3.9% | 7.5% | 78.0% | 1.12 |
| 12 h | 5.1% | 8.8% | 80.0% | 1.50 |
| 24 h | 6.3% | 9.0% | 78.9% | 1.79 |
