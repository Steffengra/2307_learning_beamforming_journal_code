
from pathlib import Path

import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim
from src.models.helpers.get_state import get_state_erroneous_channel_state_information_local

# agent 0: left: 15.1, right: 12.6
# agent 1: left: 15.5, right: 20.1


total_antennas = 16
total_users = 3

model_path = Path(
    Path(__file__).parent, '..', '..', 'models',
    '1sat_16ant_100k~0_3usr_100k~50k_additive_0.05',
    'base',
    'full_snap_2.570',
    # 'agent_1',
    'model',
)

model = load_model(model_path)

print(model.trainable_variables[0].shape)

# [radius1, radius2, ..., angle1, angle2, ...]
# [usr0ant0, usr0ant1, usr0ant2, ..., usr1ant0, ...]

sums_rad = np.zeros((total_antennas, total_users))
sums_pha = np.zeros((total_antennas, total_users))

for usr_id in range(total_users):
    for antenna_id in range(total_antennas):
        sums_rad[antenna_id, usr_id] = sum(
            abs(
                model.trainable_variables[0][usr_id*total_antennas + antenna_id, :]
            )
        )

for usr_id in range(total_users):
    for antenna_id in range(total_antennas):
        sums_pha[antenna_id, usr_id] = sum(
            abs(
                model.trainable_variables[0][usr_id*total_antennas + antenna_id + int(model.trainable_variables[0].shape[0] / 2), :]
            )
        )

# for value_id in range(model.trainable_variables[0].shape[0]):
#     if value_id == 0:
#         print('rad')
#     if value_id == model.trainable_variables[0].shape[0] / 2:
#         print('pha')
#
#     print(
#         sum(
#             abs(
#                 model.trainable_variables[0][value_id,:]
#             )
#         )
#     )


obj = sums_pha
fig, ax = plt.subplots()
im = ax.imshow(obj.T)

# Loop over data dimensions and create text annotations.
for i in range(total_antennas):
    for j in range(total_users):
        text = ax.text(i, j, round(obj[i, j], 1),
                       ha="center", va="center", color="w")

print(
    np.mean(
        sums_pha[0:5, :]
    )
)
print(
    np.mean(
        sums_pha[5:11, :]
    )
)
print(
    np.mean(
        sums_pha[11:, :]
    )
)
# print(
#     sum(
#     sum(
#         sums_pha[0:6,:]
#     ))
# )
# print(
#     sum(
#     sum(
#         sums_pha[5:11,:]
#     )
# )
# )
# print(
#     sum(
#     sum(
#         sums_pha[10:16,:]
#     )
# )
# )

plt.show()