import pytest
import torch
from flights_time_series_dataset import FlightsDataset
from time_series_models import QuantumLSTM
from time_series_predictor import TimeSeriesPredictor

MAX_EPOCHS = 50

# @pytest.mark.skip
def test_quantum_lstm_tsp_fitting():
    """
    Tests the Quantum LSTM TimeSeriesPredictor fitting
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA compatible GPU available to run this test")
    tsp = TimeSeriesPredictor(
        QuantumLSTM(),
        # lr=1E-2,
        max_epochs=MAX_EPOCHS,
        train_split=None,
        optimizer=torch.optim.Adam,
    )


    tsp.fit(FlightsDataset())
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -3

# @pytest.mark.skip
def test_quantum_lstm_tsp_fitting_in_cpu():
    """
    Tests the Quantum LSTM TimeSeriesPredictor fitting in the cpu
    """
    tsp = TimeSeriesPredictor(
        QuantumLSTM(),
        # lr=1E-2,
        max_epochs=MAX_EPOCHS,
        train_split=None,
        optimizer=torch.optim.Adam,
        device='cpu'
    )

    tsp.fit(FlightsDataset())
    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > -3