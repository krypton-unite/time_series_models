import time
from datetime import timedelta

import pytest
from torch.optim import Adam
from flights_time_series_dataset import FlightSeriesDataset
from skorch.callbacks import EarlyStopping
from skorch.dataset import CVSplit
from sklearn.metrics import r2_score  # mean_squared_error
from time_series_models import BenchmarkLSTM
from time_series_predictor import TimeSeriesPredictor

from .config import devices
from .helpers import cuda_check

@pytest.mark.parametrize('device', devices)
def test_regular(device):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    cuda_check(device)

    start = time.time()
    tsp = TimeSeriesPredictor(
        BenchmarkLSTM(hidden_dim=16),
        lr = 1e-3,
        lambda1=1e-8,
        optimizer__weight_decay=1e-8,
        iterator_train__shuffle=True,
        early_stopping=EarlyStopping(patience=50),
        max_epochs=250,
        train_split=CVSplit(10),
        optimizer=Adam,
        device=device,
    )

    past_pattern_length = 24
    future_pattern_length = 12
    pattern_length = past_pattern_length + future_pattern_length
    fsd = FlightSeriesDataset(pattern_length, past_pattern_length, pattern_length, stride=1)
    tsp.fit(fsd)
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print(f"Fitting in {device} time delta: {elapsed}")
    mean_r2_score = tsp.score(tsp.dataset)
    print(f"Achieved R2 score: {mean_r2_score}")
    assert mean_r2_score > -20