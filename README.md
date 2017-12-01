# Daily Activity IRL (DA-IRL)

## Overview:

This repository uses inverse reinforcement learning
to rationalize the full day utility function for individual
economic agents based on multiple observations of their stay sequence trajectories.

## Usage

### Input Data
It is assumed that you are have a method to extract daily activity-travel sequences.
Place traces for individual agents into a directory. 

### Running
Compatible w/ Python 2.7. 

The recommended and only currently supported installation method assumes you have the 
anaconda python scientific computing framework installed on your machine. Create
a new environment for this project and then do:

    pip install -r requirements.txt

To execute the script,  

python scripts/run_atp_experiment.py --config=data/misc/IRL_multimodal_scenario_params.json --traces_dir=<traces> --seed=1


## References


## Citation

```
@article{Feygin2017dairl,
     author = {{Feygin}, Sid and {Pozdnukhov}, Alexei},
     title = "{DA-IRL: Structural Estimation of Full Day Activity-Travel Behavior Models}",
     journal = {arXiv preprint arXiv:XXXX.XXXXXXX},
     year = 2017,
     url={https://arxiv.org/pdf/XXXX.XXXXX.pdf}
}
```