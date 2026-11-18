"""Weekly apparel demand forecasting workflow.

Required CSV columns: date, category, units_sold
Optional numeric drivers: price, promotion, temperature, google_trends

Example:
    python src/forecast_demand.py --data data/fashion_sales.csv --horizon 12
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

OPTIONAL_DRIVERS = ["price", "promotion", "temperature", "google_trends"]


def parse_args():
    parser = argparse.ArgumentParser(description="Forecast weekly apparel demand by category.")
    parser.add_argument("--data", required=True, help="Input CSV path.")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon in weeks.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    return parser.parse_args()


def load_weekly_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    raw = pd.read_csv(path)
    required = {"date", "category", "units_sold"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["units_sold"] = pd.to_numeric(raw["units_sold"], errors="coerce")
    raw["category"] = raw["category"].astype("string").str.strip()
    raw = raw.dropna(subset=["date", "category", "units_sold"])
    raw = raw[raw["category"] != ""]
    # Negative values may represent returns. Keep them in the source data but
    # aggregate them as recorded; analysts should document their return policy.
    for col in OPTIONAL_DRIVERS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    aggregations = {"units_sold": "sum"}
    for col in OPTIONAL_DRIVERS:
        if col in raw.columns:
            aggregations[col] = "max" if col == "promotion" else "mean"

    raw["week"] = raw["date"].dt.to_period("W-SUN").dt.start_time
    weekly = (
        raw.groupby(["category", "week"], as_index=False)
        .agg(aggregations)
        .rename(columns={"week": "date"})
    )

    # Reindex each category to a continuous weekly calendar. Missing sales
    # weeks remain missing, rather than silently being assumed to be zero.
    complete = []
    for category, group in weekly.groupby("category"):
        group = group.set_index("date").sort_index()
        idx = pd.date_range(group.index.min(), group.index.max(), freq="W-MON")
        group = group.reindex(idx)
        group.index.name = "date"
        group["category"] = category
        complete.append(group.reset_index())
    return pd.concat(complete, ignore_index=True).sort_values(["category", "date"])


def wape(actual, predicted) -> float:
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denominator * 100)


def make_features(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"y": series.astype(float)})
    for lag in [1, 2, 4, 8, 52]:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    for window in [4, 12]:
        frame[f"rolling_mean_{window}"] = frame["y"].shift(1).rolling(window).mean()
    frame["week_of_year"] = frame.index.isocalendar().week.astype(int)
    frame["month"] = frame.index.month
    return frame


def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    if len(train) >= 52:
        pattern = train.iloc[-52:].to_numpy()
        return np.resize(pattern, horizon).astype(float)
    if len(train) >= 4:
        pattern = train.iloc[-min(4, len(train)):].to_numpy()
        return np.resize(pattern, horizon).astype(float)
    return np.repeat(float(train.iloc[-1]), horizon)


def fit_predict_category(group: pd.DataFrame, horizon: int):
    group = group.sort_values("date").set_index("date")
    y = group["units_sold"].astype(float).asfreq("W-MON")
    y = y.interpolate(limit=2, limit_direction="both")
    y = y.dropna()
    if len(y) < 10:
        raise ValueError(
            f"Category has only {len(y)} usable weekly observations; at least 10 are required."
        )

    test_size = min(12, max(2, len(y) // 5))
    train, test = y.iloc[:-test_size], y.iloc[-test_size:]
    metrics = []
    validation_predictions = {}

    # Seasonal-naive benchmark
    naive_pred = seasonal_naive(train, len(test))
    validation_predictions["SeasonalNaive"] = naive_pred
    metrics.append(("SeasonalNaive", mean_absolute_error(test, naive_pred),
                    np.sqrt(mean_squared_error(test, naive_pred)), wape(test.to_numpy(), naive_pred)))

    # SARIMAX statistical model; seasonal component is used only with enough history.
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        seasonal_order = (1, 0, 1, 52) if len(train) >= 104 else (0, 0, 0, 0)
        model = SARIMAX(
            train, order=(1, 1, 1), seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=100)
        pred = np.asarray(fitted.forecast(len(test)), dtype=float)
        validation_predictions["SARIMAX"] = pred
        metrics.append(("SARIMAX", mean_absolute_error(test, pred),
                        np.sqrt(mean_squared_error(test, pred)), wape(test.to_numpy(), pred)))
    except Exception as exc:
        print(f"Note: SARIMAX skipped for this category ({exc}).")

    # Gradient boosting with time-aware lag features.
    try:
        train_features = make_features(train).dropna()
        test_history = train.copy()
        ml_preds = []
        if len(train_features) >= 8:
            ml_model = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                    l2_regularization=1.0, random_state=42
                )
            )
            ml_model.fit(train_features.drop(columns="y"), train_features["y"])
            for date in test.index:
                features = make_features(test_history).iloc[[-1]].drop(columns="y")
                features.index = [date]
                prediction = float(ml_model.predict(features)[0])
                ml_preds.append(prediction)
                test_history.loc[date] = float(test.loc[date])
            ml_preds = np.asarray(ml_preds)
            validation_predictions["GradientBoosting"] = ml_preds
            metrics.append(("GradientBoosting", mean_absolute_error(test, ml_preds),
                            np.sqrt(mean_squared_error(test, ml_preds)), wape(test.to_numpy(), ml_preds)))
    except Exception as exc:
        print(f"Note: Gradient boosting skipped for this category ({exc}).")

    metric_df = pd.DataFrame(metrics, columns=["model", "MAE", "RMSE", "WAPE_percent"])
    # Select the best model by WAPE; combine the best statistical/ML candidates
    # only when they beat or match the baseline.
    valid = metric_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["WAPE_percent"])
    best_name = valid.sort_values("WAPE_percent").iloc[0]["model"]
    best_validation = validation_predictions[best_name]
    residuals = np.asarray(test) - np.asarray(best_validation)
    lower_error, upper_error = np.quantile(residuals, [0.10, 0.90])

    # Refit selected model on all available data.
    future_dates = pd.date_range(y.index.max() + pd.Timedelta(weeks=1),
                                 periods=horizon, freq="W-MON")
    if best_name == "SARIMAX":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        seasonal_order = (1, 0, 1, 52) if len(y) >= 104 else (0, 0, 0, 0)
        fitted = SARIMAX(y, order=(1, 1, 1), seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False).fit(
                             disp=False, maxiter=100)
        future = np.asarray(fitted.forecast(horizon), dtype=float)
    elif best_name == "GradientBoosting":
        all_features = make_features(y).dropna()
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                          max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=42)
        )
        model.fit(all_features.drop(columns="y"), all_features["y"])
        history = y.copy()
        future_values = []
        for date in future_dates:
            x = make_features(history).iloc[[-1]].drop(columns="y")
            x.index = [date]
            value = float(model.predict(x)[0])
            future_values.append(value)
            history.loc[date] = value
        future = np.asarray(future_values)
    else:
        future = seasonal_naive(y, horizon)

    forecast = pd.DataFrame({
        "date": future_dates,
        "category": group["category"].iloc[0],
        "predicted_units": np.maximum(future, 0),
        "lower_80": np.maximum(future + lower_error, 0),
        "upper_80": np.maximum(future + upper_error, 0),
        "selected_model": best_name,
    })
    # Keep interval bounds ordered even if residual quantiles are unusual.
    forecast["lower_80"] = np.minimum(forecast["lower_80"], forecast["predicted_units"])
    forecast["upper_80"] = np.maximum(forecast["upper_80"], forecast["predicted_units"])

    metric_df["category"] = group["category"].iloc[0]
    metric_df["selected_model"] = best_name
    return forecast, metric_df, y


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:50] or "category"


def main():
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    weekly = load_weekly_data(Path(args.data))
    all_forecasts, all_metrics = [], []

    for category, group in weekly.groupby("category"):
        try:
            forecast, metrics, history = fit_predict_category(group, args.horizon)
            all_forecasts.append(forecast)
            all_metrics.append(metrics)

            plt.figure(figsize=(10, 5))
            plt.plot(history.index, history.values, label="Historical weekly units")
            plt.plot(forecast["date"], forecast["predicted_units"], marker="o",
                     label="Forecast")
            plt.fill_between(forecast["date"], forecast["lower_80"],
                             forecast["upper_80"], alpha=0.2, label="Estimated 80% range")
            plt.title(f"Apparel demand forecast: {category}")
            plt.xlabel("Week")
            plt.ylabel("Units sold")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output / f"forecast_{safe_filename(category)}.png", dpi=150)
            plt.close()
        except Exception as exc:
            print(f"Skipped category '{category}': {exc}")

    if not all_forecasts:
        raise RuntimeError(
            "No category could be modelled. Check data quality and ensure each category "
            "has at least 10 weekly observations."
        )

    pd.concat(all_forecasts, ignore_index=True).to_csv(
        output / f"forecast_{args.horizon}_weeks.csv", index=False
    )
    pd.concat(all_metrics, ignore_index=True).to_csv(
        output / "model_metrics.csv", index=False
    )
    print(f"Forecasting complete. Results saved in: {output.resolve()}")
    print("Review the metrics and plots; forecast accuracy is data-dependent.")


if __name__ == "__main__":
    main()
