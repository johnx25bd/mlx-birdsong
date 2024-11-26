# Birdsong Species Classification with Whisper

This project uses OpenAI's Whisper model to classify bird species from their songs and calls. It specifically focuses on bird recordings from Great Britain, utilizing the BirdCLEF 2024 dataset.

## Features

- Audio classification using Whisper's encoder, and various custom classification layers
- Training data visualization
- Integration with Weights & Biases for experiment tracking

## Setup

1. Install dependencies using Poetry:
```bash
poetry install
```

2. Prepare the data:
   - Download the BirdCLEF 2024 dataset
   - Unzip the files into `data/`
   - Alternatively, use the `subsample/` directory, which contains a subset of the data for faster development. (I tried to minimize modifications to the original data structure.)

## Project Structure

```
.
├── data/
│   └── birdclef-2024/      # Dataset files, needs to be downloaded from kaggle
├── notebooks/
│   ├── single.ipynb        # Training on a single example, to overfit + test pipeline
│   ├── single/             # Data + metadata for single example
│   ├── subsample.ipynb     # Training on subsample data
│   ├── subsample/          # Data + metadata for subsample dataset
│   └── data.py             # Custom dataset class
├── pyproject.toml          # Poetry dependencies
└── README.md
```

## Usage

The main training pipeline is implemented in `notebooks/subsample.ipynb`. This notebook:
- Loads and preprocesses the BirdCLEF dataset
- Filters recordings to Great Britain region
- Trains a classifier using Whisper's encoder and a custom classification head
- Logs training metrics to Weights & Biases

## Dependencies

- PyTorch
- OpenAI Whisper
- GeoPandas
- Weights & Biases
- pandas
- matplotlib
- contextily