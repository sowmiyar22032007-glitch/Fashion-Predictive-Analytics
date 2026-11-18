# Predictive Analytics and Forecasting in the Fashion Industry

**Week 4 – Junior Data Science Analyst Internship | Yuva Intern**  
**Prepared by:** Sowmiya R

## Project objective
Build a reproducible framework to forecast weekly apparel demand by product category for the next 12 weeks. The workflow follows the supplied report: define the business target, prepare public or appropriately licensed data, compare a seasonal-naive baseline with statistical and machine-learning models, validate by time, create forecasts and monitor performance.

## Files
```text
Fashion_Predictive_Analytics_Week4/
├── README.md
├── requirements.txt
├── Week4_Predictive_Analytics_Fashion_Forecasting.docx
├── data/
│   └── README.md
├── src/
│   └── forecast_demand.py
└── outputs/
    └── (forecast CSVs and charts are created here)
```

## Input data
Prepare a CSV with at least these columns:

| Column | Meaning |
|---|---|
| `date` | Transaction or sales date |
| `category` | Apparel category, such as tops, dresses or denim |
| `units_sold` | Units sold (numeric, non-negative after returns are handled) |

Optional columns: `price`, `promotion`, `temperature`, and `google_trends`. The script aggregates daily rows into weekly category totals. For optional numeric drivers, it uses weekly means; `promotion` is aggregated as the weekly maximum.

Use a dataset you are permitted to access, and document its title, publisher, source URL, access date and licence. Public dataset examples and data-source ideas are described in the report. Do not upload restricted or personal data. The repository does not contain proprietary retail data.

## Setup
Python 3.9 or later is recommended.

```bash
pip install -r requirements.txt
```

## Run
From the project root:

```bash
python src/forecast_demand.py --data data/your_fashion_sales.csv --horizon 12
```

Optional output directory:

```bash
python src/forecast_demand.py --data data/your_fashion_sales.csv --horizon 12 --output outputs
```

## Workflow implemented
1. Validate the input columns and convert dates.
2. Aggregate sales to weekly totals for each category.
3. Create lag and rolling-average features.
4. Evaluate models using a chronological holdout (not a random split).
5. Compare a seasonal-naive benchmark, SARIMAX and a gradient-boosting regression model when sufficient history is available.
6. Combine eligible model forecasts using validation error weights.
7. Estimate an 80% prediction interval from validation residuals.
8. Export forecasts, model metrics and category-level forecast charts.

The implementation uses a seasonal-naive forecast as the fallback when there is insufficient history or an optional modelling library cannot fit a category. A 52-week seasonal period is used when enough weekly history exists; otherwise the script uses a recent-history fallback.

## Outputs
- `outputs/forecast_12_weeks.csv` – category, forecast week, predicted units and 80% lower/upper bounds.
- `outputs/model_metrics.csv` – holdout MAE, RMSE, WAPE and model used per category.
- `outputs/forecast_<category>.png` – historical sales and forecast chart.

## Important interpretation notes
Forecast quality depends on the coverage and quality of the input data. The report's WAPE target is a project goal, not a guaranteed result. Compare every candidate with the seasonal-naive baseline. A forecast interval based on past validation errors is an estimate, not a guarantee. Correlation or feature importance does not prove causation. Review promotions, returns, stock-outs, unusual events and abrupt trend changes before using forecasts for business decisions.
