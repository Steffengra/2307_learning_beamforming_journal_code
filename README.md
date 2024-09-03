

# Work in progress
This code was used in the following unpublished work [(preprint not available yet)](url).

[1] Steffen Gracla, Alea Schröder, Carsten Bockelmann, Armin Dekorsy,
"Model-free Robust Beamforming in Satellite Downlink using Reinforcement Learning", under review.

Email: {**gracla**, **schroeder**, bockelmann, dekorsy}@ant.uni-bremen.de

The code version associated with this paper along with the used learned models and evaluation results is found in the releases.
The project structure is as follows
```
.
├── models                  | trained models
├── outputs
│   ├── metrics             | metrics from evaluation
├── README.md               | this file
├── requirements.txt        | project dependencies
├── src                     | python source related to..
│   ├── analysis            |   evaluation
│   ├── config              |   configuration
│   ├── data                |   data generation, i.e., satellite model
│   ├── models              |   learning models
│   ├── plotting            |   plotting
│   ├── tests               |   code tests
└── └── utils               |   shared helper functions
```