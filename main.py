import neat
import gym
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import pickle
import os

# Define a custom Keras layer for NEAT network
class NEATLayer(tf.keras.layers.Layer):
    def __init__(self, neat_network, **kwargs):
        super(NEATLayer, self).__init__(**kwargs)
        self.neat_network = neat_network
        self.neat_output = None

    def build(self, input_shape):
        self.neat_output = [self.add_weight(shape=(1,),
                                            initializer=tf.constant_initializer(0),
                                            trainable=False)
                            for _ in range(self.neat_network.num_outputs)]

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        outputs = []
        for i in range(batch_size):
            output = self.neat_network.activate(inputs[i].numpy().tolist())
            outputs.append(output)
        return tf.convert_to_tensor(outputs)

# Helper function to create environments and networks
def create_environment_and_network(client_id, variation, config):
    env = gym.make('BipedalWalker-v3')
    env.env.gravity = variation * client_id
    genome = pickle.load(open('best_genome.pkl', 'rb'))
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return env, network

# Load or create NEAT configuration
config_path = './neat_config.txt'
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        f.write("""
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
        """)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
population = neat.Population(config)

# Evaluate genomes function with RLfD (assuming demonstrations are used)
def evaluate_genomes(genomes, config, env):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        for _ in range(5):  # Average performance over multiple episodes
            state = env.reset()
            done = False
            while not done:
                action = np.clip(net.activate(state), -1, 1)
                state, reward, done, _ = env.step(action)
                fitness += reward
        genome.fitness = fitness / 5

# Check if the best genome file exists, if not, run NEAT to create it
best_genome_path = 'best_genome.pkl'
if not os.path.exists(best_genome_path):
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(lambda genomes, config: evaluate_genomes(genomes, config, create_environment_and_network(1, 1.0, config)[0]), 30)

    with open(best_genome_path, 'wb') as f:
        pickle.dump(winner, f)

# Define TensorFlow Federated model function using the NEAT genome
def model_fn():
    # Load and convert the best genome to a NEAT network
    genome = pickle.load(open(best_genome_path, 'rb'))
    neat_network = neat.nn.FeedForwardNetwork.create(genome, config)
    model = tf.keras.Sequential([
        NEATLayer(neat_network),
        tf.keras.layers.Dense(units=4, activation='tanh')  # Assuming this fits the action space
    ])
    return tff.learning.from_keras_model(
        model=model,
        input_spec=(tf.TensorSpec(shape=[None, 24], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32)),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()])

# Collect client data function
def collect_client_data(environment, net, episodes=10):
    data = []
    for _ in range(episodes):
        state = environment.reset()
        done = False
        while not done:
            action = np.clip(net.activate(state), -1, 1)
            next_state, reward, done, _ = environment.step(action)
            data.append((state, action, reward, next_state, done))
            state = next_state
    # Convert to TFF dataset format
    states, actions, rewards, next_states, dones = zip(*data)
    return tf.data.Dataset.from_tensor_slices((np.array(states), np.array(actions)))

# Initialize environments and networks for clients
clients = [create_environment_and_network(i, 1.0 + 0.1 * i, config) for i in range(5)]

# Federated Learning Trainer Setup
trainer = tff.learning.build_federated_averaging_process(model_fn)
state = trainer.initialize()

for round_num in range(10):
    # Collect data for each client
    client_data = [collect_client_data(client[0], client[1]) for client in clients]
    state, metrics = trainer.next(state, client_data)
    print(f'Round {round_num} metrics:', metrics)

# Implementing RLfD (create or load demonstrations)
demo_file = 'demonstrations.pkl'

def create_demonstrations(env, num_demos=10):
    demos = []
    for _ in range(num_demos):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Replace with expert policy if available
            next_state, _, done, _ = env.step(action)
            demos.append((state, action))
            state = next_state
    with open(demo_file, 'wb') as f:
        pickle.dump(demos, f)

def load_demonstrations():
    with open(demo_file, 'rb') as f:
        return pickle.load(f)

# Create or load demonstrations
if not os.path.exists(demo_file):
    env_demo = gym.make('BipedalWalker-v3')
    create_demonstrations(env_demo)

demonstrations = load_demonstrations()

# Example function to use demonstrations in NEAT evaluation
def evaluate_with_demos(genomes, config, env, demonstrations):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        for demo in demonstrations:
            state, action = demo
            predicted_action = net.activate(state)
            fitness -= np.linalg.norm(predicted_action - action)  # Penalize for deviation from demonstration
        genome.fitness = fitness

# Modify NEAT running to include RLfD
winner = population.run(lambda genomes, config: evaluate_with_demos(genomes, config, create_environment_and_network(1, 1.0, config)[0], demonstrations), 30)

# Save the best genome for initialization
with open(best_genome_path, 'wb') as f:
    pickle.dump(winner, f)

# Rerun federated learning with new best genome after RLfD
state = trainer.initialize()
for round_num in range(10):
    client_data = [collect_client_data(client[0], neat.nn.FeedForwardNetwork.create(winner, config)) for client in clients]
    state, metrics = trainer.next(state, client_data)
    print(f'Round {round_num} metrics:', metrics)
