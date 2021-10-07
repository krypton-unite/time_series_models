# Time series models

[![PyPI version](https://badge.fury.io/py/time-series-models.svg)](https://badge.fury.io/py/time-series-models) [![travis](https://app.travis-ci.com/krypton-unite/time_series_models.svg?branch=master)](https://app.travis-ci.com/github/krypton-unite/time_series_models) [![codecov](https://codecov.io/gh/krypton-unite/time_series_models/branch/master/graph/badge.svg)](https://codecov.io/gh/krypton-unite/time-series-models) [![GitHub license](https://img.shields.io/github/license/krypton-unite/time_series_models)](https://github.com/krypton-unite/time_series_models) [![Requirements Status](https://requires.io/github/krypton-unite/time_series_models/requirements.svg?branch=master)](https://requires.io/github/krypton-unite/time_series_models/requirements/?branch=master)

## Description
Time series neural network models for [Time series predictor](https://github.com/krypton-unite/time_series_predictor)

## Installation

```terminal
pip install time-series-models
```

## Usage example

```python
from time_series_models import BenchmarkLSTM
from skorch.callbacks import EarlyStopping
from skorch.dataset import CVSplit
from torch.optim import Adam
from flights_time_series_dataset import FlightSeriesDataset
from time_series_predictor import TimeSeriesPredictor

tsp = TimeSeriesPredictor(
    BenchmarkLSTM(),
    lr = 1e-3,
    lambda1=1e-8,
    optimizer__weight_decay=1e-8,
    iterator_train__shuffle=True,
    early_stopping=EarlyStopping(patience=50),
    max_epochs=250,
    train_split=CVSplit(10),
    optimizer=Adam
)

past_pattern_length = 24
future_pattern_length = 12
pattern_length = past_pattern_length + future_pattern_length
fsd = FlightSeriesDataset(pattern_length, past_pattern_length, pattern_length, stride=1)
tsp.fit(fsd)
mean_r2_score = tsp.score(tsp.dataset)
print(f"Achieved R2 score: {mean_r2_score}")
assert mean_r2_score > -20
```