# mtpcm
Multi-task Proteochemometric Modelling

## Installation

Create the envoronment using yml file and install PCM packega in it:
```
conda env create --file pcm_env.yml
pip install -e .
```

## Usage

All required classes (models + dataloaders/modules) are provided in `PCM/` folder. 
Classes `PCM` and `PCM_ext` correspond to basic PCM and PCM-ext models discussed in the manuscript. 
Classes `PCM_MT` and `PCM_MT_withPRT` correspond to basic MT and MT-PCM models discussed in the manuscript.

`example.py` provides minimal code for using basic models using dummy synthetic data.
`example_optuna.py` demonstrates how Optuna can be used in conjunction with PCM classes.
