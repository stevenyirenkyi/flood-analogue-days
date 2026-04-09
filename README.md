# Identifying Flood-Analogue Days for AI Model Development in Data-Scarce Coastal Environments

## Overview

This repository contains the code and data accompanying the paper:

The workflow identifies flood-analogue days (days whose multivariate coastal forcing
conditions resemble those of documented flood events) using globally available
reanalysis products. It requires no local gauge networks or dense observational records,
making it applicable to data-scarce coastal environments.

The three-phase workflow:

1. Constructs the Dynamic Coastal Forcing Index (DCFI) from four oceanographic drivers
2. Identifies extreme forcing days using semi-supervised k-means clustering
3. Refines the extreme day set using a compound wind-wave filter

---

<!--
## Repository Structure
flood-analogue-days/
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── harmonised_data/                                # Harmonised datasets from Phase 1
│   ├── flood_occurence_year_month.csv                  # Flood reports obtained from NADMO, online news reports, and social media (Table A4)
│   └── Flood Event Dates in Keta-Anloga Municipal.pdf  # Official flood event reports obtained from NADMO
├── utils/                                              # utilities for calculating dcfi, managing paths, loading data, etc
├── data_harmonisaton.ipynb/                            # data harmonisation
├── phase_2.py                                          # K-means clustering and extreme day detection
├── phase_3.py                                          # Wind-wave characterisation and filter and flood-analogue days
├── FloodAnalyzer.py                                    # Functionalities for clusterings and cluster analysis -->

## Data

### Provided in this repository

| File                                                  | Description                                             |
| ----------------------------------------------------- | ------------------------------------------------------- |
| `data/harmonised_data/`                               | Preprocessed driver data ready for Phase 2 and 3        |
| `data/flood_occurence_year_month.csv`                 | Flood reports from NADMO, online news, and social media |
| `data/Flood Event Dates in Keta-Anloga Municipal.pdf` | Official NADMO flood event documentation                |

### Raw source data (not included)

The raw reanalysis and model datasets used in Phase 1 are publicly available from their
respective providers. Due to file size and provider terms, they are not redistributed here.

| Driver                               | Source  | Access                             |
| ------------------------------------ | ------- | ---------------------------------- |
| Tide (T)                             | FES2022 | https://www.aviso.altimetry.fr/    |
| Wave run-up (R)                      | ERA5    | https://cds.climate.copernicus.eu/ |
| Dynamic atmospheric correction (DAC) | Mog2D   | https://www.aviso.altimetry.fr/    |
| Sea level anomaly (SLA)              | ORAS5   | https://cds.climate.copernicus.eu/ |

All datasets are freely accessible upon registration with the respective services.

---
