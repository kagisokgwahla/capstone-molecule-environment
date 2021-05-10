```
              _                 _      
  /\/\   ___ | | ___  ___ _   _| | ___ 
 /    \ / _ \| |/ _ \/ __| | | | |/ _ \
/ /\/\ \ (_) | |  __/ (__| |_| | |  __/
\/    \/\___/|_|\___|\___|\__,_|_|\___|
   __           _                                      _   
  /__\ ____   _(_)_ __ ___  _ __  _ __ ___   ___ _ __ | |_ 
 /_\| '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|
//__| | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_ 
\__/|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__|

```                                                        
![testing](https://github.com/robmacc/capstone-molecule-environment/workflows/testing/badge.svg)

Reinforcement learning environment for inverse drug design.

## Set-up

Install dependencies:
```
conda env create -f environment.yml
```
Activate environment:
```
conda activate mol-env
```

## Getting started
To get a working gym environment all that's needed is to use the provided repository structure 
(see [here](https://github.com/openai/gym/blob/master/docs/creating-environments.md)):

* Any dependencies that the environment needs must be defined in `setup.py`. 
* The environment's entry point must be defined in `gym_molecule/__init__.py`
* The environment needs to be imported into `gym_molecule/envs/__init.py__`
* With this structure the environment can be installed with `pip install -e .`
from the working directory.
* The environment definition must be written in `gym_molecule/envs/molecule_env`,
and should implement the interface provided by the `gym.Env` class (see 
the definition [here](https://github.com/openai/gym/blob/master/gym/core.py)).
* The essential methods which need definitions are `step, reset, render, seed,`
and `close`.
* These stubs have been provided in gym_molecule/envs/molecule_env.py.


## Testing
Please use pytest to test the environment, an example test file 
(see `tests/environment_test`) and a testing workflow script
(see `.github/workflows/testing`) have been provided.
