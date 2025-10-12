### this theorist replaces our previous hard_concrete deep learning implementation; while less impressive, it's easier to document and integrate with the grading scheme
# import bms, autora state, on_state
from autora.theorist.bms import BMSRegressor
import pandas as pd
from autora.state import on_state
import ray

class ParallelizedAkaikeBMSRegressorPool:
    """
    A wrapper around BMSRegressor that parallelizes fitting multiple instances using Ray.
    It chooses the best model based on approximate akaike critetion after fitting. Otherwise it behaves like a standard BMSRegressor.
    """
    def __init__(self, epochs=100, pool_size=12):
        """
        Arguments:
            epochs: Number of epochs to train each BMSRegressor instance.
            pool_size: Number of parallel BMSRegressor instances to fit.
        """
        self.epochs = epochs
        self.pool_size = pool_size
        self.best_model  = None


    def fit(self, experiment_data, variables):
        """
        Fits multiple BMSRegressor instances in parallel and selects the best one.
        Arguments (from state):
            experiment_data: A pandas DataFrame containing the experimental data.
            variables: A VariableCollection containing independent and dependent variable definitions.
        Returns:
            self: The fitted ParallelizedBMSRegressor instance with the best model selected.

        """
        independent_var_names = [var.name for var in variables.independent_variables]
        dependent_var_names = [var.name for var in variables.dependent_variables]
        X = experiment_data[independent_var_names]
        #to numPy array
        X = X.to_numpy()
        y = experiment_data[dependent_var_names]
        y = y.to_numpy()
        #print("Fitting ParallelizedAkaikeBMSRegressorPool with data X:", X, "and y:", y)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Define a remote function to fit a single BMSRegressor, remote meaning its run in a separate process via Ray
        @ray.remote
        def fit_single_bms(X, y, epochs):
            try:
                bms = BMSRegressor(epochs=epochs)
                bms.fit(X, y)
                return bms
            except Exception as e:
                print(f"Error fitting BMSRegressor: {e}")
                return None

        futures = [fit_single_bms.remote(X, y, self.epochs) for _ in range(self.pool_size)]
        results = ray.get(futures)
        print(f'Failed fits: {len([res for res in results if res is None])} out of {self.pool_size}')
        results = [res for res in results if res is not None]

        # Select the model with the lowest approximate akaike criterion
        def compute_aic(bms):
            k = len(bms.repr().split(' '))
            return 2 * k + bms.loss_
        self.best_model = min(results, key=compute_aic)
        return [self.best_model]

def main():
    # experiment_runner
    from autora.experiment_runner.synthetic.psychophysics.weber_fechner_law import weber_fechner_law
    from autora.experiment_runner.synthetic.psychophysics.stevens_power_law import stevens_power_law
    from autora.experiment_runner.synthetic.economics.expected_value_theory import expected_value_theory
    from autora.state import StandardState, on_state


    # data handling
    from sklearn.model_selection import train_test_split



    def benchmark(experiment_runner, theorist):
        # DEMONSTRATION: FITTING Part 1
        @on_state(output=['models'])
        def theorist_on_state(experiment_data, variables):
            benchmark_theorist = theorist.fit(experiment_data, variables)
            return benchmark_theorist

        conditions = experiment_runner.domain()
        # generate all corresponding observations
        experiment_data = experiment_runner.run(conditions, added_noise=0.001)
        state = StandardState(variables=experiment_runner.variables, conditions=conditions, experiment_data=experiment_data, models=[])
        #DEMONSTRATION: FITTING Part 2 (theorist_on_state function is called here)
        state = theorist_on_state(state)
        benchmark_theorist = state.models[-1]
        # evaluate theorist performance
        independent_var_names = [var.name for var in experiment_runner.variables.independent_variables]
        dependent_var_names = [var.name for var in experiment_runner.variables.dependent_variables]
        X = state.experiment_data[independent_var_names]
        y = state.experiment_data[dependent_var_names]
        #DEMONSTRATION: PREDICTION
        y_pred = benchmark_theorist.predict(X)
        error = ((y - y_pred) ** 2).mean().values[0]
        print(f"Theorist: {theorist.__class__.__name__}, Validation MSE: {error}")
        #DEMONSTRATION: Obtaining the equation
        winning_equation = benchmark_theorist.repr()
        print(f"Winning Equation: {winning_equation}")



    #%%
    # from autora.theorist.autora_theorist_polynomial import ExampleRegressor

    my_theorist = ParallelizedAkaikeBMSRegressorPool(epochs=1500, pool_size=24)
    #my_theorist = BMSRegressor(epochs=10)  # simpler baseline
    #my_theorist = dummy_theorist # remove this once you implemented your own theorist
    print("Benchmarking Stevens' Power Law")
    benchmark(experiment_runner=stevens_power_law(), theorist=my_theorist)
    print("Benchmarking Weber-Fechner Law")
    benchmark(experiment_runner=weber_fechner_law(), theorist=my_theorist)
    print("Benchmarking Expected Value Theory")
    benchmark(experiment_runner=expected_value_theory(), theorist=my_theorist)


if __name__ == "__main__":
    main()
