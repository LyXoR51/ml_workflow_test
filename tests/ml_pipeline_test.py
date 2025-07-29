import pytest
from unittest import mock
from app.train import load_data, preprocess_data, create_pipeline, train_model

# Test data loading
def test_load_data():
    url = "https://fp-private-bucket.s3.eu-west-3.amazonaws.com/housing_prices/real_estate_dataset.csv"
    df = load_data(url)
    assert not df.empty, "Dataframe is empty"

# Test data preprocessing
def test_preprocess_data():
    df = load_data("https://fp-private-bucket.s3.eu-west-3.amazonaws.com/housing_prices/real_estate_dataset.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"

# Test pipeline creation
def test_create_pipeline():
    pipe = create_pipeline()
    assert "Preprocessing" in pipe.named_steps, "Preprocessing missing in pipeline"
    assert "Regressor" in pipe.named_steps, "LinearRegression missing in pipeline"

# Test model training ()
@mock.patch('app.train.LinearRegression.fit', return_value=None)
def test_train_model(mock_fit):
    pipe = create_pipeline()
    X_train, X_test, y_train, y_test = preprocess_data(load_data("https://fp-private-bucket.s3.eu-west-3.amazonaws.com/housing_prices/real_estate_dataset.csv"))
    model = train_model(pipe, X_train, y_train)
    assert model is not None, "Model training failed"
