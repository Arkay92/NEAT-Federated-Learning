# NEAT-Federated-Learning

NEAT-Federated combines the power of NeuroEvolution of Augmenting Topologies (NEAT) with TensorFlow Federated (TFF) for distributed reinforcement learning. This project evolves neural networks using NEAT and utilizes federated learning to train models across multiple clients in a decentralized manner.

## Features

- **NEAT Integration**: Evolve neural networks using the NEAT algorithm.
- **Federated Learning**: Train models across multiple clients without sharing raw data.
- **Custom Keras Layer**: Implement NEAT networks as custom Keras layers.
- **Gym Environment**: Use Gym environments for reinforcement learning tasks.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
Initialize NEAT Population:

Ensure your NEAT configuration file (neat_config.txt) is properly set up.
Run the NEAT algorithm to evolve the neural networks and save the best genome.
Federated Learning:

- Use the saved genome to initialize client models.
- Simulate federated learning across multiple clients to train the models.

## Example Code

### NEAT Configuration (neat_config.txt)
```
[NEAT]
fitness_criterion     = max
fitness_threshold     = 300
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
num_hidden            = 0
num_inputs            = 24
num_outputs           = 4
initial_connection    = full
activation_default    = tanh
activation_mutate_rate= 0.1
activation_options    = tanh relu sigmoid
node_add_prob         = 0.2
node_delete_prob      = 0.1
conn_add_prob         = 0.3
conn_delete_prob      = 0.2
conn_enable_rate      = 0.25
conn_disable_rate     = 0.25

[DefaultReproduction]
elitism               = 2
survival_threshold    = 0.2

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func  = max
max_stagnation        = 20
species_elitism       = 2

[SteadyState]
replacement_rate      = 0.2
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
