import time
from datetime import timedelta

import pytest
import torch
from flights_time_series_dataset import FlightsDataset
from time_series_models import QuantumLSTM
from time_series_predictor import TimeSeriesPredictor

MAX_EPOCHS = 50

# @pytest.mark.skip
# @pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_quantum_lstm_tsp_fitting(device='cuda'):
    """
    Tests the Quantum LSTM TimeSeriesPredictor fitting
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA compatible GPU available to run this test")
    tsp = TimeSeriesPredictor(
        QuantumLSTM(),
        lr=1E-1,
        max_epochs=50,
        train_split=None,
        optimizer=torch.optim.Adam,
        device=device
    )

    start = time.time()
    tsp.fit(FlightsDataset())
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in {} time delta: {}".format(device, elapsed))
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -3
