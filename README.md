# Daily Activity IRL (DA-IRL)

**NOTICE**: _This repository is currently in pre-alpha testing and should not be used in production for any purpose, as it is subject
to change rapidly and without backwards compatibility. There are additional dependencies that have not yet been
made public as well._ 

_This code is currently provided mainly for research purposes and ongoing work with collaborators._

## Overview:

This repository uses inverse reinforcement learning
to rationalize the full day utility function for individual
economic agents based on multiple observations of their stay sequence trajectories.

## Usage

### Input Data
It is assumed that you are have a method to extract daily activity-travel sequences.
Place traces for individual agents into a directory. 

### Running
Compatible w/ Python 2 and 3.

The recommended and only currently supported installation method assumes you have the 
anaconda python scientific computing framework installed on your machine. Create
a new environment for this project and then do:

    pip install -r requirements.txt

To execute the test script  

    python scripts/run_atp_experiment.py --config=data/misc/IRL_multimodal_scenario_params.json --traces_dir=<TRACES> --seed=1

where `<TRACES>` is a directory of persona trace files each ending in .csv.

## References


## Citation

 You may use the following BibTeX entry to cite this work in a publication (**TO BE UPDATED**):

```
@article{Feygin2017dairl,
     author = {{Feygin}, Sid and {Pozdnukhov}, Alexei},
     title = "{DA-IRL: Structural Estimation of Full Day Activity-Travel Behavior Models}",
     journal = {arXiv preprint arXiv:XXXX.XXXXXXX},
     year = 2017,
     url={https://arxiv.org/pdf/XXXX.XXXXX.pdf}
}
```