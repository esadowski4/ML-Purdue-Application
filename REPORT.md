# TSA Passenger Forecasting - Beginner Application - Eric Sadowski

Before I start the report, I just wanted to thank whoever is reading this for giving me the opportunity to complete this assessment. Whether or not I am picked to be on the project, I learned a ton from this takehome assessment and just wanted to say thank you, as I am now even more interested in learning as much as I can about machine learning!

## Methodology

I used a **Gradient Boosting Regressor** from scikit-learn to forecast daily TSA passenger volumes. I learned that Gradient Boosting works by building many small decision trees sequentially, where each new tree tries to correct the mistakes of the previous ones. I chose this over a Random Forest because Gradient Boosting tends to perform better on structured tabular data like the data I was given.

### Model Parameters
- `n_estimators=500` - builds 500 trees for more learning opportunities
- `max_depth=4` - limits each tree's complexity to avoid overfitting
- `learning_rate=0.05` - each tree contributes a small correction, leading to more stable results
- `min_samples_leaf=10` - each leaf must have at least 10 data points, preventing the model from memorizing noise
- `random_state=42` - ensures the same results every time the code is run

### Features Used

**Calendar features** (extracted from the date):
- `day_of_week` - 0 (Monday) through 6 (Sunday)
- `day_of_month` - 1 through 31
- `month` - 1 through 12
- `day_of_year` - 1 through 365
- `year` - captures the overall growth trend in passenger volume
- `is_weekend` - binary flag for Saturday/Sunday

**Lag features** (historical passenger volumes):
- `lag7` - passenger volume from 7 days ago (same day last week)
- `lag14` - volume from 14 days ago
- `lag28` - volume from 28 days ago (same day ~4 weeks ago)
- `lag365` - volume from 365 days ago (same day last year)

The lag features turned out to be by far the most important. `lag7` alone accounted for roughly 65% of the model's predictive power, which makes intuitive sense - a Tuesday this week will look a lot like last Tuesday.

### Future Features (If I Had Time)

I actually had a ton of ideas about what features I could add, but didn't know how to (I'm fairly new to ML in general), and due to having very little time to work on this (very busy week), I was not able to. Here are some of the ideas I had:

- `political events` - People are less likely to travel during war uncertainty or if they feel unsafe (such as war with Iran, reports of planes falling out of the sky, the government being closed and air traffic controllers aren't working, etc.)
- `holidays` - this is pretty self explanatory, and is probably already implemented in the model you guys have
- `large-scale events` - This was a little wildcard, but lots of people travel to certain places for things like the Olympics, Super Bowl, or World Cup (which is happening this year)
- `politics in general` - There could be new laws passed that prohibit certain people from visiting certain places (such as the middle east), which could affect travelers
- `economic factors` - If our economy is in a recession (which I think will be very soon lol), much less people will travel I believe

## Data Analysis

### How I Handled the Data

1. **Sorted by date** to ensure chronological order, which is essential for time series data.
2. **Extracted calendar features** from the Date column using pandas `.dt` accessor.
3. **Built lag features** by creating a date-to-volume lookup from training data, then mapping each row's past dates to their historical volumes.
4. **Dropped rows with NaN lags** - the first ~365 rows couldn't have a `lag365` value since that would require data from before the training set starts. After dropping, about 913 training rows remained.
5. **Validation split** - held out the last 60 days of training as a validation set to test model accuracy before making final predictions. This simulates the real task of predicting future dates.
6. **Retrained on full training data** - after confirming validation performance, I retrained the model using all training rows so the final predictions benefit from the most recent data.
7. **Handled test NaN lags** - some test dates had lag lookups that fell outside the training data range. I used forward fill to handle these missing values.


## Learnings

The most challenging aspects of building this baseline model were:

1. **Understanding lag features.** The biggest conceptual hurdle for me was understanding what lag features were and how to implement them. While implementing them, I ran into issues with losing some data to train on. I had to figure out how to deal with this problem (I ended up using dropna and ffill).

2. **The NaN tradeoff.** Creating `lag365` meant losing an entire year of training data (rows where a year-ago lookup doesn't exist). The feature is powerful, but it costs training rows. Despite losing data, the model improved after adding it.

3. **Iterating on features.** My first attempt with only calendar features gave an MAE of ~89,000. Adding lag features initially made it slightly worse (~95,000) because of lost training rows, but after adding `lag365` and tuning hyperparameters, it dropped to ~83,000. This taught me that feature engineering is iterative - not every addition helps immediately.


## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python model.py
```

This will train the model, print validation metrics, generate test predictions, and save a plot (`predictions_plot.png`) showing predicted vs actual values.