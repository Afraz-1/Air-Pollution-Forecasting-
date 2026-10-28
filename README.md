# Beijing PM2.5: one-hour-ahead forecasting

This project forecasts the next hour's PM2.5 concentration from the preceding 24 hours of pollution and weather observations. It compares a small LSTM with two simple persistence forecasts and Ridge regression. The reported results come from a chronological holdout: train on 2010–2012, select the LSTM checkpoint on 2013, and evaluate once on 2014.

The corrected [`Air quality forecasting.ipynb`](Air%20quality%20forecasting.ipynb) runs the experiment and displays its metrics and diagnostic plots. The original exploratory notebook is preserved in [`archive/`](archive/). Its saved metrics used a test CSV that is also contained in its validation period, so **do not use those metrics as an independent test result**. `train.py` contains the shared training and evaluation code.

## Data

The files are available from [Air Pollution Forecasting - LSTM Multivariate on Kaggle](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate). The hourly observations derive from the [UCI Beijing PM2.5 dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) (2010–2014, US Embassy PM2.5 measurements and Beijing airport weather). The target is PM2.5 in µg/m³. The dataset is historical and is suitable for method comparison, not a current air-quality service.

Run `download_data.py` to place `LSTM-Multivariate_pollution.csv` and `pollution_test_data1.csv` in `data/`. The training script uses only the first file. **The separate test file is excluded**: all 346 of its records are the final 346 rows of the main CSV and therefore cannot form a separate holdout. `data/` is ignored by Git; download the files locally to reproduce the experiment.

## Run

Use Python 3.12. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe train.py
```

To run the notebook, launch JupyterLab with `.\.venv\Scripts\python.exe -m jupyter lab` and open `Air quality forecasting.ipynb`. Running its cells retrains the models and regenerates the figures.

The script saves `outputs/metrics.json`, `outputs/test_predictions.csv`, `outputs/best_lstm.keras`, `outputs/ridge.joblib`, and `outputs/scalers.joblib`. The output directory is ignored by Git. For a quick experiment, `python train.py --epochs 5` is supported; the metrics below use the default 12-epoch limit and early stopping.

## Evaluation design

- **Target:** PM2.5 at hour *t*.
- **Inputs:** observations from hours *t−24* through *t−1*: pollution, dew point, temperature, pressure, wind speed, snow, rain, wind direction, and cyclical hour/day-of-year features. No observation at hour *t* enters its prediction.
- **Splits:** 26,256 training targets through 2012; 8,760 validation targets in 2013; 8,760 untouched test targets in 2014. Windows at a split boundary may use earlier observed history, as they would during live one-hour-ahead forecasting.
- **Preprocessing:** feature and target scalers are fitted only on rows before 2013. The LSTM checkpoint is selected by 2013 validation Huber loss. Seeds are fixed at 42 and TensorFlow deterministic operations are enabled.
- **Comparisons:** persistence predicts the previous hour; seasonal persistence predicts the same hour one day earlier; Ridge uses the same flattened 24-hour input window as the LSTM. Ridge alpha is fixed at 1.0.

## Measured 2014 test results

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² |
| --- | ---: | ---: | ---: |
| Previous hour | 12.4298 | 24.5147 | 0.9314 |
| Previous day, same hour | 67.9166 | 100.1784 | -0.1463 |
| Ridge | 12.9155 | **23.7451** | **0.9356** |
| LSTM | 13.6148 | 24.0143 | 0.9341 |

The previous-hour forecast has the lowest MAE; Ridge has the lowest RMSE and highest R². This LSTM does not outperform those baselines on the held-out year. These are one-hour-ahead results using observed recent pollution, so they should not be presented as multi-hour or weather-only forecasts. See [`results/metrics.json`](results/metrics.json) for the recorded run's train and validation metrics, versions, and split counts; a fresh run writes `outputs/metrics.json`.

The archived notebook's screenshot shows **MAE 41.7099, MSE 3301.3106, RMSE 57.4570, and R² 0.6626**. It evaluated 298 windows from a duplicate 346-row CSV and used weather history without past pollution as an input. The corrected experiment evaluates 8,760 hours in 2014 using past pollution and weather. These results are for different evaluation sets and inputs; the numerical change is not evidence that the LSTM itself improved.

## Limitations

The data come from one city and end in 2014. A single 2014 holdout measures one historical period; rolling-origin evaluation and a newer dataset would give stronger evidence of generalization. TensorFlow results can vary slightly across hardware or library versions even with fixed seeds.
