import pandas as pd
from autora.state import VariableCollection
from autora.variable import IV, DV
from akaike_bms import ParallelizedAkaikeBMSRegressorPool

def test_parallelized_akaike_bms_regressor_pool_fit():
    # Create dummy data
    X = pd.DataFrame({'x1': [1, 2, 3, 4], 'x2': [2, 3, 4, 5]})
    y = pd.DataFrame({'y': [2, 4, 6, 8]})
    experiment_data = pd.concat([X, y], axis=1)

    # Create dummy variables
    variables = VariableCollection(
        independent_variables=[IV(name='x1'), IV(name='x2')],
        dependent_variables=[DV(name='y')]
    )

    # Instantiate and fit
    theorist = ParallelizedAkaikeBMSRegressorPool(epochs=10, pool_size=2)
    models = theorist.fit(experiment_data, variables)

    # Assert that a model is returned
    assert models is not None
    assert len(models) == 1
    assert hasattr(models[0], 'predict')


def test_parallelized_akaike_bms_regressor_pool_predict_and_repr():
    # Synthetic data
    X = pd.DataFrame({'x1': [0, 1, 2, 3], 'x2': [1, 2, 3, 4]})
    y = pd.DataFrame({'y': [1, 3, 5, 7]})
    experiment_data = pd.concat([X, y], axis=1)

    variables = VariableCollection(
        independent_variables=[IV(name='x1'), IV(name='x2')],
        dependent_variables=[DV(name='y')]
    )

    theorist = ParallelizedAkaikeBMSRegressorPool(epochs=5, pool_size=2)
    models = theorist.fit(experiment_data, variables)
    model = models[0]

    # Predict
    y_pred = model.predict(X)
    assert y_pred.shape == (4, 1)

    # Check equation representation
    eq = model.repr()
    assert isinstance(eq, str)
    assert len(eq) > 0

def main():
    test_parallelized_akaike_bms_regressor_pool_fit()
    test_parallelized_akaike_bms_regressor_pool_predict_and_repr()
    print("All tests passed.")
if __name__ == "__main__":
    main()