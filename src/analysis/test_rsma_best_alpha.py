
import numpy as np
import matplotlib.pyplot as plt

from src.config.config import Config
from src.analysis.helpers.test_rsma_precoder import test_rsma_precoder_user_distance_sweep


def main():

    rsma_factors = np.arange(0, 1+0.1, step=0.1)  # exclusive interval
    common_part_precoding_style = 'MRT'
    monte_carlo_iterations = 100

    distance_sweep_range = np.linspace(0, 50_000, 100)

    cfg = Config()
    cfg.show_plots = False

    results = np.zeros((len(rsma_factors), len(distance_sweep_range)))

    for rsma_factor_id, rsma_factor in enumerate(rsma_factors):

        metrics = test_rsma_precoder_user_distance_sweep(
            config=cfg,
            distance_sweep_range=distance_sweep_range,
            rsma_factor=rsma_factor,
            common_part_precoding_style=common_part_precoding_style,
            monte_carlo_iterations=monte_carlo_iterations,
        )

        results[rsma_factor_id, :] = metrics[list(metrics.keys())[0]]['mean']

    fig, ax = plt.subplots()
    indices = ax.twinx()

    best_values = np.max(results, axis=0)
    best_indices = np.argmax(results, axis=0)

    ax.plot(distance_sweep_range, best_values, label='sum rate')
    indices.scatter(distance_sweep_range, rsma_factors[best_indices], label='rsma alpha', color='black')

    fig.legend(labelcolor="linecolor")

    plt.show()


if __name__ == '__main__':
    main()
