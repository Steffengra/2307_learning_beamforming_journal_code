
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle

from src.config.config import Config
from src.analysis.helpers.test_rsma_precoder import test_rsma_precoder_user_distance_sweep



def main():

    rsma_factors = np.arange(0, 1+0.1, step=0.1)  # exclusive interval
    common_part_precoding_style = 'basic'
    monte_carlo_iterations = 10000

    distance_sweep_range = np.linspace(0, 50_000, 100)

    cfg = Config()
    cfg.show_plots = False

    results = np.zeros((len(rsma_factors), len(distance_sweep_range)))
    results_fairness = np.zeros((len(rsma_factors), len(distance_sweep_range)))

    for rsma_factor_id, rsma_factor in enumerate(rsma_factors):

        metrics = test_rsma_precoder_user_distance_sweep(
            config=cfg,
            distance_sweep_range=distance_sweep_range,
            rsma_factor=rsma_factor,
            common_part_precoding_style=common_part_precoding_style,
            monte_carlo_iterations=monte_carlo_iterations,
            metrics=['sumrate','fairness']
        )

        results[rsma_factor_id, :] = metrics[list(metrics.keys())[0]]['mean']
        results_fairness[rsma_factor_id, :] = metrics[list(metrics.keys())[1]]['mean']

    print(results.shape)
    print(results_fairness.shape)
    fig, ax = plt.subplots()
    indices = ax.twinx()

    best_values = np.max(results, axis=0)
    best_indices = np.argmax(results, axis=0)
    fairness_at_best = results_fairness[best_indices, np.arange(results.shape[1])]
    best_power_factor = rsma_factors[best_indices]
    print(best_values.shape)
    print(fairness_at_best.shape)
    # fairness_at_best_indices = results_fairness[:,best_indices]
    # print(fairness_at_best_indices)
    # print(fairness_at_best_indices.shape)
    # print(best_values.shape)
    ax.plot(distance_sweep_range, best_values, label='sum rate')
    ax.plot(distance_sweep_range, fairness_at_best, label='fairness')
    indices.scatter(distance_sweep_range, rsma_factors[best_indices], label='rsma alpha', color='black')

    fig.legend(labelcolor="linecolor")



    plt.show()

    metrics = {'calc_sum_rate': {'mean': best_values}, 'fairness at best sumrate': fairness_at_best, 'power factor': best_power_factor}

    # print(best_values)
    precoder_name = 'rsma_genie'

    def save_results():
        name = (
            f'testing_{precoder_name}'
            f'_sweep_{distance_sweep_range[0]}_{distance_sweep_range[-1]}'
            f'.gzip'
        )
        results_path = Path(cfg.output_metrics_path, cfg.config_learner.training_name, 'user_distance_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump([distance_sweep_range, metrics], file=file)

    save_results()

if __name__ == '__main__':
    main()
