This repo contains a set of algorithms to design coating stacks for gravitational wave detectors mirror coatings. There is a mixture of reinforcement learning, genetic algorithms and mcmc methods which are compared.

There are a number of available algorithms to run
 - DQN
 - HPPO
 - HPPO OML ** recommended **
 - genetic algorithm
 - MCMC

Installation

Create a conda environment for this project (or any other environment of your choosing). Its only been tested on python 3.11

```bash
conda create -n coatings_optimisation python=3.11
```

```bash
    $pip install .
``` 

Run instructions (for HPPO OML)

```bash 
    $python -m coatopt.train_hppo_oml -c ./config.ini --train
```

There are example configs in the /examples/ folder, this also includes the choice of materials in a json file. The above script will then put a set of outputs into the chosen root directory in the config file

You should see an evolution of the reward and values similar to below
![rewards](https://raw.githubusercontent.com/jcbayley/coatopt/refs/heads/main/examples/running_values.png)

Where the optimal coating stack from this run looks like:
![rewards](https://raw.githubusercontent.com/jcbayley/coatopt/refs/heads/main/examples/best_state.png)
