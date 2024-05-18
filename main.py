import neat, gym, pickle, os
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a custom Keras layer for NEAT network
class NEATLayer(tf.keras.layers.Layer):
    def __init__(self, neat_network, **kwargs):
        super(NEATLayer, self).__init__(**kwargs)
        self.neat_network = neat_network

    def call(self, inputs):
        outputs = tf.vectorized_map(lambda x: tf.convert_to_tensor(self.neat_network.activate(x.numpy().tolist())), inputs)
        return outputs

# Helper function to create environments and networks for each client
# Load NEAT configuration and create environments and networks
def create_environment_and_network(client_id, variation, config):
    env = gym.make('BipedalWalker-v3')
    env.env.gravity = variation * client_id
    genome = load_genome('best_genome.pkl')
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return env, network

def create_neat_config(path):
    config_content = """
   [NEAT]
fitness_criterion = max
fitness_threshold = 300
pop_size = 150
reset_on_extinction = False

[DefaultGenome]
num_hidden = 0
num_inputs = 24
num_outputs = 4
initial_connection = full

activation_default = tanh
activation_mutate_rate = 0.1
activation_options = tanh relu sigmoid

aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum

bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1

response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5

conn_add_prob = 0.5
conn_delete_prob = 0.5

enabled_default = True
enabled_mutate_rate = 0.01

feed_forward = True

node_add_prob = 0.2
node_delete_prob = 0.2

weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30
weight_min_value = -30
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 20
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2

[SteadyState]
replacement_rate = 0.2
    """
    with open(path, 'w') as config_file:
        config_file.write(config_content.strip())

config_path = './neat_config.txt'
if not os.path.exists(config_path):
    print(f"Configuration file '{config_path}' not found. Creating default configuration file.")
    create_neat_config(config_path)
else:
    # Overwrite the existing file to ensure it's correct
    print("Configuration file exists. Overwriting with correct format.")
    create_neat_config(config_path)

try:
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
except Exception as e:
    print(f"An error occurred while loading the NEAT configuration: {e}")
    raise

# Evaluate genomes function with reinforcement learning from demonstrations (RLfD), Parallel processing of genomes evaluation
def evaluate_genome(genome, config, env):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    for _ in range(5):
        state = env.reset()
        done = False
        while not done:
            action = np.clip(net.activate(state), -1, 1)
            state, reward, done, _ = env.step(action)
            fitness += reward
    return fitness / 5, genome

def evaluate_genomes(genomes, config, env):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluate_genome, genome, config, env) for _, genome in genomes]
        for future in concurrent.futures.as_completed(futures):
            fitness, genome = future.result()
            genome.fitness = fitness

# NEAT training and saving the best genome if not already saved
best_genome_path = 'best_genome.pkl'
population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

if not os.path.exists(best_genome_path):
    winner = population.run(lambda genomes, config: evaluate_genomes(genomes, config, gym.make("BipedalWalker-v3")), 50)
    with open(best_genome_path, 'wb') as f:
        pickle.dump(winner, f)

# TensorFlow Federated model function using the NEAT genome
def model_fn():
    genome = load_genome('best_genome.pkl')
    config_path = './neat_config.txt'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    neat_network = neat.nn.FeedForwardNetwork.create(genome, config)
    model = tf.keras.Sequential([
        NEATLayer(neat_network),
        tf.keras.layers.Dense(units=4, activation='tanh')
    ])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=(tf.TensorSpec(shape=[None, 24], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32)),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()])

# Collect client data for federated learning using Gym environments
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
    states, actions, rewards, next_states, dones = zip(*data)
    dataset = tf.data.Dataset.from_tensor_slices((np.array(states), np.array(actions)))
    return dataset.batch(32)

# Create demonstrations for reinforcement learning, saved as pickle
demo_file = 'demonstrations.pkl'
if not os.path.exists(demo_file):
    env_demo = gym.make('BipedalWalker-v3')
    demos = []
    for _ in range(10):  # Create 10 demonstrations
        state = env_demo.reset()
        done = False
        while not done:
            action = env_demo.action_space.sample()  # Random actions as placeholders
            next_state, _, done, _ = env_demo.step(action)
            demos.append((state, action))
            state = next_state
    with open(demo_file, 'wb') as f:
        pickle.dump(demos, f)

demonstrations = pickle.load(open(demo_file, 'rb'))

def load_genome(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File '{filepath}' not found. Exiting.")
        raise SystemExit

# Evaluate genomes with RLfD using demonstrations
def evaluate_with_demos(genomes, config, env, demonstrations):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        for _ in range(5):
            state = env.reset()
            done = False
            while not done:
                action = np.clip(net.activate(state), -1, 1)
                state, reward, done, _ = env.step(action)
                fitness += reward
        genome.fitness = fitness / 5

# NEAT training incorporating demonstrations
winner = population.run(lambda genomes, config: evaluate_with_demos(genomes, config, create_environment_and_network(1, 1.0, config)[0], demonstrations), 30)
with open(best_genome_path, 'wb') as f:
    pickle.dump(winner, f)

# Load simulation data for federated learning
source, _ = tff.simulation.datasets.emnist.load_data()

def client_data(n):
    source, _ = tff.simulation.datasets.emnist.load_data()
    return source.create_tf_dataset_for_client(source.client_ids[n]).map(
        lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
    ).repeat(10).batch(20)

# Pick a subset of client devices to participate in training
train_data = [client_data(n) for n in range(3)]

# Wrap a Keras model for use with TensorFlow Federated
# Federated Learning Execution
tff_model = model_fn()
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    tff_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

state = trainer.initialize()

for _ in range(5):
    result = trainer.next(state, train_data)
    state = result.state
    metrics = result.metrics
    logger.info(metrics['client_work']['train']['accuracy'])

def federated_train(num_clients, num_rounds):
    clients = [create_environment_and_network(i, 1.0 + 0.1 * i, config) for i in range(num_clients)]
    train_data = [client_data(n) for n in range(num_clients)]

    model = model_fn()
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))
    state = trainer.initialize()

    for round_num in range(num_rounds):
        result = trainer.next(state, train_data)
        state = result.state
        metrics = result.metrics
        logger.info(f'Round {round_num + 1}: {metrics["client_work"]["train"]["accuracy"]}')
    
    return state, metrics

# Federated Learning Testing and Benchmarking Class
class FederatedLearningTest:
    def __init__(self, clients, model_fn, trainer, state, config, demonstrations):
        self.clients = clients
        self.model_fn = model_fn
        self.trainer = trainer
        self.state = state
        self.config = config
        self.demonstrations = demonstrations

    def run_federated_training(self, rounds=10):
        metrics_list = []
        for round_num in range(rounds):
            client_data = [collect_client_data(client[0], client[1]) for client in self.clients]
            result = self.trainer.next(self.state, client_data)
            self.state = result.state
            metrics_list.append(result.metrics)
            logger.info(f'Round {round_num} metrics: {result.metrics}')
        return metrics_list

    def evaluate_model(self, env, network, episodes=10):
        total_reward = 0
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = np.clip(network.activate(state), -1, 1)
                state, reward, done, _ = env.step(action)
                total_reward += reward
        return total_reward / episodes

    def plot_metrics(self, metrics_list):
        rounds = range(len(metrics_list))
        mse = [metrics['mean_squared_error'].numpy() for metrics in metrics_list]

        plt.figure(figsize=(10, 5))
        plt.plot(rounds, mse, label='Mean Squared Error')
        plt.xlabel('Rounds')
        plt.ylabel('Mean Squared Error')
        plt.title('Federated Learning Training Metrics')
        plt.legend()
        plt.show()

    def plot_rewards(self, rewards):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(rewards)), rewards, label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Model Rewards Over Episodes')
        plt.legend()
        plt.show()

    def benchmark(self, baseline_reward):
        neat_network = neat.nn.FeedForwardNetwork.create(pickle.load(open(best_genome_path, 'rb')), self.config)
        client_rewards = [self.evaluate_model(client[0], neat_network) for client in self.clients]
        avg_reward = np.mean(client_rewards)

        logger.info(f'Average reward after federated learning: {avg_reward}')
        logger.info(f'Baseline reward: {baseline_reward}')
        
        plt.figure(figsize=(10, 5))
        plt.bar(['Baseline', 'Federated Learning'], [baseline_reward, avg_reward])
        plt.ylabel('Average Reward')
        plt.title('Benchmarking')
        plt.show()

# Initialize environments and networks for clients
clients = [create_environment_and_network(i, 1.0 + 0.1 * i, config) for i in range(5)]

# Federated Learning Trainer Setup
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.legacy.SGD(learning_rate=0.1))
state = trainer.initialize()

# Testing and benchmarking
test = FederatedLearningTest(clients, model_fn, trainer, state, config, demonstrations)

# Run federated training
state, final_metrics = federated_train(5, 10)

# Plot training metrics
test.plot_metrics(metrics)

# Evaluate and plot rewards
neat_network = neat.nn.FeedForwardNetwork.create(pickle.load(open(best_genome_path, 'rb')), config)
rewards = [test.evaluate_model(client[0], neat_network) for client in clients]
test.plot_rewards(rewards)

# Benchmark the model
baseline_reward = 100  # Replace with actual baseline reward for comparison
test.benchmark(baseline_reward)
