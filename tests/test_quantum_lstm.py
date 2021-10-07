import time
from datetime import timedelta

import pytest
import numpy as np
from torch.optim import Adam
from flights_time_series_dataset import FlightSeriesDataset, FlightsDataset
from skorch.callbacks import EarlyStopping
from skorch.dataset import CVSplit
from sklearn.metrics import r2_score  # mean_squared_error
from time_series_models import QuantumLSTM
from time_series_predictor import TimeSeriesPredictor

from .config import devices
from .helpers import cuda_check

@pytest.mark.skip
@pytest.mark.parametrize('device', devices)
def test_regular(device):
    """
    Tests the LSTMTimeSeriesPredictor fitting
    """
    cuda_check(device)

    start = time.time()
    tsp = TimeSeriesPredictor(
        QuantumLSTM(),
        lr = 1e-1,
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
    assert mean_r2_score > -10

@pytest.mark.parametrize('device', devices)
def test_quantum_lstm_tsp_fitting(device):
    """
    Tests the Quantum LSTM TimeSeriesPredictor fitting
    """
    cuda_check(device)

    tsp = TimeSeriesPredictor(
        QuantumLSTM(),
        lr=1E-1,
        max_epochs=50,
        train_split=None,
        optimizer=Adam,
        device=device
    )

    start = time.time()
    tsp.fit(FlightsDataset())
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in {} time delta: {}".format(device, elapsed))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -10

@pytest.mark.parametrize('device', devices)
def test_quantum_lstm_tsp_forecast(device):
    """
    Tests the Quantum LSTM forecast
    """
    cuda_check(device)

    tsp = TimeSeriesPredictor(
        QuantumLSTM(hidden_dim = 2),
        max_epochs=250,
        lr = 1e-4,
        early_stopping=EarlyStopping(patience=100, monitor='train_loss'),
        train_split=None,
        optimizer=Adam,
        device=device
    )

    whole_fd = FlightsDataset()
    # leave last N months for error assertion
    last_n = 24
    start = time.time()
    tsp.fit(FlightsDataset(pattern_length = 120, except_last_n = last_n))
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print(f"Fitting in {device} time delta: {elapsed}")
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -5

    netout, _ = tsp.forecast(last_n)

    # Select any training example just for comparison
    idx = np.random.randint(0, len(tsp.dataset))
    _, whole_y = whole_fd[idx]

    y_true = whole_y[-last_n:, :]   # get only known future outputs
    y_pred = netout[idx, -last_n:, :]    # get only last N predicted outputs
    r2s = r2_score(y_true, y_pred)
    assert r2s > -60
