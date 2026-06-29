from neuroglyph_data.synthetic import synthetic_session, write_synthetic_processed


def test_synthetic_epochs_shape():
    X, y = synthetic_session(n_events=64, task="hand")
    assert X.shape[1:] == (14, 25)
    assert y.shape[0] == X.shape[0]
    assert y.min() >= 0
    assert y.max() < 2


def test_write_processed(tmp_path):
    path = write_synthetic_processed(tmp_path, n_events=32, task="zone")
    assert path.exists()