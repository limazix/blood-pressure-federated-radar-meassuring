# blood-pressure-federated-radar-meassuring

# Requirements
1. Python (v3.8)
2. Poetry (v1.12)

# Dataset
This code relies on a publicly available dataset described [here](https://www.nature.com/articles/s41597-020-00629-5). It contains labelled contact sensor data alongside radar data, which are all labelled with 5 possible scenarios: Valsalva maneuver, Apnea simulation, Tilt up and Tilt down movements (on a tilting table) and Resting. It totals 24h worth of signals coming from 30 subjects.

The data should be placed in a directory at the project level called data. Inside, it should contain one directory to each subject named by its id.

# Usage

## Setup

```sh
poetry install
```

## Run

```sh
poetry run python scripts/svm_single_subject.py
```
