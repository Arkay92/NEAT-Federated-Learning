import warnings
import neat
import gym
import pickle
import os
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import matplotlib.pyplot as plt
import logging
import concurrent.futures
import cProfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants
CONFIG_PATH = './neat_config.txt'
BEST_GENOME_PATH = './best_genome.pkl'
DEMO_FILE = './demonstrations.pkl'
NUM_CLIENTS = 5
MAX_GEN = 3 
NUM_ROUNDS = 10
EPISODES_PER_EVALUATION = 15
MAX_EPISODES = 500
NUM_INPUTS = 24
NUM_OUTPUTS = 4

class NEATLayer(tf.keras.layers.Layer):
    def __init__(self, neat_network, **kwargs):
        super(NEATLayer, self).__init__(**kwargs)
        self.neat_network = neat_network

    def call(self, inputs):
        outputs = tf.vectorized_map(lambda x: tf.convert_to_tensor(self.neat_network.activate(x.numpy().tolist())), inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4)  # Assuming the NEAT network outputs 4 values

class FederatedLearningTest:
    def __init__(self, clients, model_fn, trainer, state, config, demonstrations):
        self.clients = clients
        self.model_fn = model_fn
        self.trainer = trainer
        self.state = state
        self.config = config
        self.demonstrations = demonstrations

    def run_federated_training(self, rounds=NUM_ROUNDS):
        metrics_list = []
        for round_num in range(rounds):
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                client_data = list(executor.map(lambda client: collect_client_data(client[0], client[1]), self.clients))
            result = self.trainer.next(self.state, client_data)
            self.state = result.state
            metrics_list.append(result.metrics)
            logger.info(f'Round {round_num + 1} metrics: {result.metrics}')
        return metrics_list

    def evaluate_model(self, env, network, episodes=MAX_GEN):
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
        mse = [metrics['client_work']['train']['mean_squared_error'] for metrics in metrics_list]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, mse, label='Mean Squared Error')
        plt.xlabel('Rounds')
        plt.ylabel('Mean Squared Error')
        plt.title('Federated Learning Training Metrics')
        plt.legend()
        plt.show()

    def plot_rewards(self, rewards, label):
        plt.figure(figsize=(10, 3))
        plt.plot(range(len(rewards)), rewards, label=label)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title(f'Model Rewards Over Episodes ({label})')
        plt.legend()
        plt.show()

    def benchmark(self, baseline_reward):
        neat_network = neat.nn.FeedForwardNetwork.create(pickle.load(open(BEST_GENOME_PATH, 'rb')), self.config)
        client_rewards = [self.evaluate_model(client[0], neat_network) for client in self.clients]
        avg_reward = np.mean(client_rewards)

        logger.info(f'Average reward after federated learning: {avg_reward}')
        logger.info(f'Baseline reward: {baseline_reward}')

        plt.figure(figsize=(10, 3))
        plt.bar(['Baseline', 'Federated Learning'], [baseline_reward, avg_reward])
        plt.ylabel('Average Reward')
        plt.title('Benchmarking')
        plt.show()

    def plot_client_performance(self, client_rewards):
        plt.figure(figsize=(10, 3))
        for i, rewards in enumerate(client_rewards):
            plt.plot(range(len(rewards)), rewards, label=f'Client {i+1}')
        plt.xlabel('Rounds')
        plt.ylabel('Rewards')
        plt.title('Federated Learning Client Performance')
        plt.legend()
        plt.show()

# Helper function to create environments and networks for each client
def create_environment_and_network(client_id, variation, config):
    env = gym.make('BipedalWalker-v3')
    env._max_episode_steps = MAX_EPISODES  # Reduce max episode steps to speed up
    env.env.gravity = variation * client_id
    genome = load_genome(BEST_GENOME_PATH)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return env, network

def create_neat_config(path):
    config_content = """
    [NEAT]
    fitness_criterion = max
    fitness_threshold = 300
    pop_size = 50
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
    logger.info(f"Created new NEAT configuration file at {path}")

# Check and create configuration file if needed
create_neat_config(CONFIG_PATH)

try:
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
except Exception as e:
    logger.error(f"Failed to load NEAT configuration: {e}")
    exit(1)

def evaluate_genome(genome_id, genome, config):
    env = gym.make('BipedalWalker-v3')
    env._max_episode_steps = MAX_EPISODES  # Ensure max episode steps are set
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    try:
        for _ in range(EPISODES_PER_EVALUATION):
            state = env.reset()
            done = False
            while not done:
                action = np.clip(net.activate(state), -1, 1)
                state, reward, done, _ = env.step(action)
                fitness += reward
    except gym.error.Error as e:
        logger.error(f"Gym environment error during genome {genome_id} evaluation: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during genome {genome_id} evaluation: {e}")
    finally:
        env.close()
    return fitness / EPISODES_PER_EVALUATION, genome_id

def evaluate_genomes(genomes, config):
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(evaluate_genome, genome_id, genome, config): genome_id for genome_id, genome in genomes}
        for future in concurrent.futures.as_completed(futures):
            genome_id = futures[future]
            try:
                fitness, genome_id = future.result()
                genome = next(genome for gid, genome in genomes if gid == genome_id)
                genome.fitness = fitness
            except Exception as e:
                logger.error(f"Error evaluating genome {genome_id}: {e}")

def model_fn():
    genome = load_genome(BEST_GENOME_PATH)
    neat_network = neat.nn.FeedForwardNetwork.create(genome, config)
    hidden_weights, output_weights, hidden_biases, output_biases = extract_neat_weights(neat_network, NUM_INPUTS, NUM_OUTPUTS)
    model = create_tf_model_from_neat(NUM_INPUTS, NUM_OUTPUTS, hidden_weights, output_weights, hidden_biases, output_biases)

    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=(
            tf.TensorSpec(shape=[None, NUM_INPUTS], dtype=tf.float32),
            tf.TensorSpec(shape=[None, NUM_OUTPUTS], dtype=tf.float32)
        ),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()])

def collect_client_data(environment, net, episodes=MAX_GEN):
    data = []
    try:
        for _ in range(episodes):
            state = environment.reset()
            done = False
            while not done:
                action = np.clip(net.activate(state), -1, 1)
                next_state, reward, done, _ = environment.step(action)
                data.append((state, action))
                state = next_state
        states, actions = zip(*data)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((states, actions)).batch(32)
        return dataset
    except Exception as e:
        logger.error(f"Error collecting client data: {e}")
        return None

if not os.path.exists(DEMO_FILE):
    env_demo = gym.make('BipedalWalker-v3')
    demos = []
    for _ in range(5):  # Reduce number of demonstrations to speed up
        state = env_demo.reset()
        done = False
        while not done:
            action = env_demo.action_space.sample()  # Random actions as placeholders
            next_state, _, done, _ = env_demo.step(action)
            demos.append((state, action))
            state = next_state
    with open(DEMO_FILE, 'wb') as f:
        pickle.dump(demos, f)
demonstrations = pickle.load(open(DEMO_FILE, 'rb'))

def load_genome(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File '{filepath}' not found. Exiting.")
        raise SystemExit

def evaluate_with_demos(genomes, config, env, demonstrations):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        for _ in range(EPISODES_PER_EVALUATION):
            state = env.reset()
            done = False
            while not done:
                action = np.clip(net.activate(state), -1, 1)
                state, reward, done, _ = env.step(action)
                fitness += reward
        genome.fitness = fitness / EPISODES_PER_EVALUATION

def federated_train(num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS):
    clients = [create_environment_and_network(i, 1.0 + 0.1 * i, config) for i in range(num_clients)]
    train_data = [collect_client_data(client[0], client[1]) for client in clients]
    model = model_fn()
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    )
    state = trainer.initialize()
    metrics_list = []
    for round_num in range(num_rounds):
        result = trainer.next(state, train_data)
        state = result.state
        metrics = result.metrics
        metrics_list.append(metrics)
        logger.info(f'Round {round_num + 1}: {metrics["client_work"]["train"]["mean_squared_error"]}')

    return state, metrics_list

def train_neat_non_federated(config, generations=MAX_GEN):
    env = gym.make('BipedalWalker-v3')
    env._max_episode_steps = MAX_EPISODES
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    for generation in range(generations):
        # Evaluate the population
        genomes = list(population.population.items())
        logger.info(f'Starting generation {generation + 1}/{generations}')
        population.run(lambda genomes, config: evaluate_genomes(genomes, config), 1)
        best_genome = population.best_genome
        if best_genome.fitness >= config.fitness_threshold:
            logger.info(f'Early stopping at generation {generation + 1}')
            break

    return population.best_genome, stats

def plot_comparison(federated_rewards, non_federated_rewards):
    plt.figure(figsize=(10, 5))
    episodes = range(len(federated_rewards))
    plt.plot(episodes, federated_rewards, label='Federated NEAT')
    plt.plot(episodes, non_federated_rewards, label='Non-Federated NEAT')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Comparison of Federated and Non-Federated NEAT')
    plt.legend()
    plt.show()

def extract_neat_weights(neat_network, num_inputs, num_outputs):
    # Determine the number of hidden nodes by excluding input and output nodes
    total_nodes = len(neat_network.node_evals)
    num_hidden = total_nodes - num_outputs - num_inputs

    # Initialize weight matrices and bias vectors
    hidden_weights = np.zeros((num_inputs, num_hidden)) if num_hidden > 0 else np.array([])
    output_weights = np.zeros((num_hidden if num_hidden > 0 else num_inputs, num_outputs))
    hidden_biases = np.zeros(num_hidden) if num_hidden > 0 else np.array([])
    output_biases = np.zeros(num_outputs)

    # Track hidden node indices to map correctly
    hidden_indices = {}

    # Populate the hidden_indices dictionary
    for node in neat_network.node_evals:
        node_index = node[0]
        if num_inputs <= node_index < num_inputs + num_hidden:
            hidden_indices[node_index] = node_index - num_inputs

    # Fill weights and biases from connections
    for node in neat_network.node_evals:
        node_index, activation, aggregation, bias, response, inputs = node[:6]
        if node_index in hidden_indices:  # It's a hidden node
            hidden_biases[hidden_indices[node_index]] = bias
            for input_id, weight in inputs:
                if input_id < num_inputs:  # Connection from input to this hidden node
                    hidden_weights[input_id, hidden_indices[node_index]] = weight
                else:  # Connection from another hidden node
                    if input_id in hidden_indices:
                        hidden_weights[hidden_indices[input_id], hidden_indices[node_index]] = weight
        elif node_index >= num_inputs + num_hidden:  # It's an output node
            output_index = node_index - (num_inputs + num_hidden)
            if 0 <= output_index < num_outputs:  # Ensure the index is within bounds
                output_biases[output_index] = bias
                for input_id, weight in inputs:
                    if input_id in hidden_indices:  # Connection from a hidden node to this output node
                        output_weights[hidden_indices[input_id], output_index] = weight
                    elif input_id < num_inputs:  # Connection from an input node directly to output
                        output_weights[input_id, output_index] = weight

    return hidden_weights, output_weights, hidden_biases, output_biases

def create_tf_model_from_neat(num_inputs, num_outputs, hidden_weights, output_weights, hidden_biases, output_biases):
    inputs = tf.keras.Input(shape=(num_inputs,))
    x = inputs
    if hidden_weights.size > 0:
        x = tf.keras.layers.Dense(
            units=hidden_weights.shape[1], activation='relu', use_bias=True,
            kernel_initializer=tf.keras.initializers.Constant(hidden_weights),
            bias_initializer=tf.keras.initializers.Constant(hidden_biases)
        )(x)
    x = tf.keras.layers.Dense(
        units=num_outputs, activation='tanh', use_bias=True,
        kernel_initializer=tf.keras.initializers.Constant(output_weights),
        bias_initializer=tf.keras.initializers.Constant(output_biases)
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

if __name__ == "__main__":
    # Profile the script to find bottlenecks
    profiler = cProfile.Profile()
    profiler.enable()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)

    # Create a population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Evaluate the population
    genomes = list(population.population.items())
    
    if not os.path.exists(BEST_GENOME_PATH):
        winner = population.run(lambda genomes, config: evaluate_genomes(genomes, config), MAX_GEN)
        with open(BEST_GENOME_PATH, 'wb') as f:
            pickle.dump(winner, f)

    winner = pickle.load(open(BEST_GENOME_PATH, 'rb'))
    winner, stats = train_neat_non_federated(config, MAX_GEN)
    with open(BEST_GENOME_PATH, 'wb') as f:
        pickle.dump(winner, f)

    clients = [create_environment_and_network(i, 1.0 + 0.1 * i, config) for i in range(NUM_CLIENTS)]
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    )
    state = trainer.initialize()

    test = FederatedLearningTest(clients, model_fn, trainer, state, config, demonstrations)
    state, federated_metrics = federated_train(NUM_CLIENTS, NUM_ROUNDS)

    neat_network = neat.nn.FeedForwardNetwork.create(pickle.load(open(BEST_GENOME_PATH, 'rb')), config)
    federated_rewards = [test.evaluate_model(client[0], neat_network) for client in clients]
    non_federated_rewards = [test.evaluate_model(client[0], neat_network, MAX_GEN) for client in clients]

    plot_comparison(federated_rewards, non_federated_rewards)

    client_rewards = []
    for client in clients:
        client_rewards.append([test.evaluate_model(client[0], neat_network) for _ in range(NUM_ROUNDS)])
    test.plot_client_performance(client_rewards)

    test.plot_metrics(federated_metrics)

    # Disable profiler and print profiling results
    profiler.disable()
    profiler.print_stats(sort='cumtime')