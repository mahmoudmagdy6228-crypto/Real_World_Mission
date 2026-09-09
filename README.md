# E-Commerce Sales Prediction & Data Drift Detection

## Overview

This project explores a real-world machine learning scenario where an e-commerce sales prediction model starts producing inaccurate predictions because customer behavior and market conditions change over time.

The main goal is not only to build a prediction model, but also to detect **Data Drift**, identify when model performance starts to deteriorate, and define an appropriate operational response.

## Dataset

The dataset contains raw e-commerce sales data covering the period from **November 1, 2025 to January 9, 2026**.

* **Rows:** 1,459
* **Time period:** 70 days
* **Target:** `units_sold`

### Features

| Feature             | Description              |
| ------------------- | ------------------------ |
| `date`              | Transaction date         |
| `day_of_week`       | Day of the week          |
| `product_category`  | Product category         |
| `price`             | Product price            |
| `discount_pct`      | Discount percentage      |
| `is_holiday_season` | Holiday season indicator |
| `units_sold`        | Number of units sold     |

## Project Workflow

The project follows a time-based machine learning workflow:

1. Load and inspect the dataset
2. Perform basic data cleaning and validation
3. Convert the date column to datetime
4. Create calendar-based features
5. Split the data chronologically
6. Train a Linear Regression model
7. Generate sales predictions
8. Evaluate the model using MAE
9. Monitor MAE week by week
10. Detect the point where model performance significantly deteriorates
11. Investigate the root cause
12. Define an operational response and retraining strategy

## Model

A **Linear Regression** model was trained using:

* `price`
* `discount_pct`
* `is_holiday_season`

The model was trained only on the first part of the timeline and evaluated on later observations to simulate a real production scenario.

### Time-Based Split

**Training:**

* November 1, 2025 → November 21, 2025
* 442 observations

**Testing:**

* November 22, 2025 → January 9, 2026
* 1,017 observations

A chronological split was used instead of a random train/test split because this is a time-dependent prediction problem.

## Model Performance

The overall test MAE was approximately:

**13.73 units**

However, the overall MAE hides an important change in model performance over time.

### Weekly MAE

| Test Week |   MAE |
| --------: | ----: |
|         1 |  4.30 |
|         2 |  4.09 |
|         3 |  3.68 |
|         4 |  4.24 |
|         5 | 23.37 |
|         6 | 28.14 |
|         7 | 28.11 |
|         8 | 25.68 |

The model performs consistently during the first four weeks, with MAE around 4 units.

Starting from **Week 5**, the error increases dramatically.

## Data Drift Detection

The stable reference MAE from Weeks 1–4 was approximately:

**4.08**

An alert threshold was defined as:

**2 × Reference MAE = 8.15**

Week 5 exceeded this threshold, making it the first detected alert period.

The first alert started on:

**December 19, 2025**

## Root Cause

The investigation indicates that the performance degradation is caused by **Data Drift rather than a software bug**.

The holiday season began on **December 15, 2025**, and customer purchasing behavior changed significantly.

More importantly, the training period contained no holiday-season observations. Therefore, the model had no examples from which to learn the effect of the holiday regime.

This means that simply adding the `is_holiday_season` feature does not solve the problem if the training data contains no variation in that feature.

## Operational Response

The recommended response is:

### 1. Contain the incident

When the seven-day MAE exceeds twice the stable reference level, stop using the legacy model for automated purchasing or staffing decisions.

### 2. Use a fallback

Temporarily use a reviewed seasonal fallback forecast while the model is being investigated and updated.

### 3. Collect new data

Continue collecting labeled observations from the new seasonal regime, including sales, prices, discounts, product categories, and holiday/event information.

### 4. Retrain safely

After sufficient holiday-season data becomes available, train a new model using:

* Product category information
* Calendar features
* Event/holiday information
* Recent observations

### 5. Validate before deployment

Use time-ordered validation and compare the new model against the legacy model on recent data.

The new model should only be deployed if it meets the agreed business performance threshold and improves meaningfully over the legacy model.

### 6. Prevent recurrence

Monitor model performance continuously using weekly and daily MAE and trigger alerts when performance exceeds the defined threshold.

## Key Takeaways

* A model can work well during training and still fail when real-world conditions change.
* Monitoring model performance over time is essential.
* Overall metrics can hide important periods of model failure.
* Data Drift is not necessarily a coding bug.
* Retraining blindly on the original data can reproduce the same problem.
* Production ML systems need monitoring, fallback strategies, and controlled retraining.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Linear Regression
* MAE
* Data Drift Monitoring

## Project Structure

```text
E-Commerce-Sales-Drift/
│
├── ML_case.ipynb
├── ecommerce_sales_raw.csv
└── README.md
```

## Conclusion

This project demonstrates a practical machine learning scenario where model performance deteriorates because the underlying data distribution changes.

The main lesson is that building a model is only one part of an ML system. **Monitoring, detecting drift, responding to incidents, and retraining safely are equally important.**
