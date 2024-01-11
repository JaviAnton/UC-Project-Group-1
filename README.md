# UC-Course-Project-Group-1 Extracting k-primary traffic corridors from sparse trajectory samplings

## Overview
This is the implementation of the course project for Urban Computing. We propose a pipeline model for extracting leading corridors. First, we generate synthetic sparse trajectory data based on the road network and real-world bike counter data. Afterwards, the travel time distribution of the road network is updated using the data to recover trajectories. Finally, Row-wise Similarity matrix and k-Medoids clustering algorithm are applied on the recovered trajectories to determine the leading corridors.

## Getting Started

### Prerequisites
- python3

### Usage

To run this project, please follow these steps:

1. **Bike Counter Data Download**
    Download the Paris Bike Counter Data from here and paste it to the data folder: https://www.data.gouv.fr/fr/datasets/comptage-velo-donnees-compteurs/
    
2. **Generate Trajectories**
    Run "DataGenerator.py", this file contains the synthetic data generation algorithm.

3. **Run Experiments**
    Run "main.ipynb" for the route recovery and trajectory clustering.

* The "Archived" folder contains all the code and data we have implemented during the project, but not used in the final version of the project.

## Contact
- Óscar Nebreda Bernal: s3434745@vuw.leidenuniv.nl
- Shaoxuan Zhang: s3426505@vuw.leidenuniv.nl