
import numpy as np
import matplotlib.pyplot as plt

from src.config.config_plotting import generic_styling
from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim
from src.data.calc_sum_rate_RSMA import calc_sum_rate_RSMA
from src.utils.get_precoding import get_precoding_rsma


def main():

    rsma_factors = np.arange(0, 1, step=0.1)

    cfg = Config()
    sat_man = SatelliteManager(config=cfg)
    usr_man = UserManager(config=cfg)

    update_sim(cfg, sat_man, usr_man)

    sum_rates = np.zeros(len(rsma_factors))

    for rsma_idx, rsma_factor in enumerate(rsma_factors):
        rsma_precoding_matrix = get_precoding_rsma(
            config=cfg,
            satellite_manager=sat_man,
            rsma_factor=rsma_factor,
            common_part_precoding_style='basic',
        )

        sum_rates[rsma_idx] = calc_sum_rate_RSMA(
            channel_state=sat_man.channel_state_information,
            w_precoder=rsma_precoding_matrix,
            noise_power_watt=cfg.noise_power_watt
        )

    fig, ax = plt.subplots()
    ax.plot(rsma_factors, sum_rates)
    generic_styling(ax)
    plt.show()


if __name__ == '__main__':
    main()
