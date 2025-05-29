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
from sklearn.metrics import mean_squared_error
from tensorflow_addons.layers import GroupNormalization

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants
CONFIG_PATH = './neat_config.txt'
BEST_GENOME_PATH = './best_genome.pkl'
DEMO_FILE = './demonstrations.pkl'
NUM_CLIENTS = 10
MAX_GEN = 5
NUM_ROUNDS = 5
EPISODES_PER_EVALUATION = 25
MAX_EPISODES = 500
NUM_INPUTS = 24
NUM_OUTPUTS = 4
CHECKPOINT_DIR = './checkpoints'

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def precompute_neat_outputs(env, network, num_samples):
    inputs = []
    outputs = []
    for _ in range(num_samples):
        state = env.reset()
        done = False
        while not done:
            action = np.clip(network.activate(state), -1, 1)
            inputs.append(state)
            outputs.append(action)
            state, _, done, _ = env.step(action)
    return np.array(inputs), np.array(outputs)

# Modify the NEATMarkovLayer to accept precomputed outputs
class NEATMarkovLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NEATMarkovLayer, self).__init__(**kwargs)
        self.gp_model = self._create_gp_model()

    def _create_gp_model(self):
        # Define a simple sequential model as the internal model
        return tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Assuming the output is a single scalar
        ])

    def call(self, inputs):
        return self.gp_model(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)  # Assuming the final output is a single value

# Federated Learning Test
class FederatedLearningTest:
    def __init__(self, clients, model_fn, trainer, state, config, demonstrations):
        self.clients = clients
        self.model_fn = model_fn
        self.trainer = trainer
        self.state = state
        self.config = config
        self.demonstrations = demonstrations
        self.client_learning_rates = self.adjust_learning_rates()

    def run_federated_training(self, rounds=NUM_ROUNDS):
      metrics_list = []
      for round_num in range(rounds):
          self.client_learning_rates = self.adjust_learning_rates()  # Adjust learning rates each round
          with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
              client_data = list(executor.map(lambda client: list(collect_client_data(client[0], client[1], self.client_learning_rates[client[2]])), self.clients))
          weights = [self.evaluate_model(client[0], client[1]) for client in self.clients]
          total_weight = sum(weights)
          self.normalized_weights = [weight / total_weight for weight in weights]
          aggregated_data = self.weighted_aggregation(client_data)
          result = self.trainer.next(self.state, aggregated_data)
          self.state = result.state
          metrics_list.append(result.metrics)
          logger.info(f'Round {round_num + 1} metrics: {result.metrics}')
          self.save_checkpoint(round_num)
          if self.early_stopping(metrics_list):
              logger.info(f'Early stopping at round {round_num + 1}')
              break
      return self.state, metrics_list

    def adjust_learning_rates(self):
        client_learning_rates = {}
        for i, client in enumerate(self.clients):
            performance = self.evaluate_model(client[0], client[1])
            learning_rate = 0.1 / (1 + np.exp(-performance))  # Example adjustment based on performance
            client_learning_rates[i] = learning_rate
            logger.info(f'Client {i} learning rate adjusted to {learning_rate}')
        return client_learning_rates

    def early_stopping(self, metrics_list, threshold=0.01, patience=3):
        if len(metrics_list) < patience:
            return False
        recent_mse = [metrics['client_work']['train']['mean_squared_error'] for metrics in metrics_list[-patience:]]
        return np.mean(recent_mse) < threshold

    def weighted_aggregation(self, client_data):
      def aggregate_batches(batch_list, weights):
          max_size = max(batch['x'].shape[0] for batch in batch_list)
          agg_x = np.zeros((max_size, batch_list[0]['x'].shape[1]), dtype=np.float32)
          agg_y = np.zeros((max_size, batch_list[0]['y'].shape[1]), dtype=np.float32)

          for batch, weight in zip(batch_list, weights):
              x_padded = np.pad(batch['x'].numpy(), ((0, max_size - batch['x'].shape[0]), (0, 0)), 'constant')
              y_padded = np.pad(batch['y'].numpy(), ((0, max_size - batch['y'].shape[0]), (0, 0)), 'constant')
              agg_x += x_padded * weight
              agg_y += y_padded * weight

          return {'x': agg_x, 'y': agg_y}

      all_batches = list(zip(*[list(client_data[i]) for i in range(len(client_data))]))
      aggregated_data = []
      for batch_group in all_batches:
          batch_dicts = [batch for batch in batch_group]
          aggregated_batch = aggregate_batches(batch_dicts, self.normalized_weights)
          aggregated_data.append(tf.data.Dataset.from_tensor_slices(aggregated_batch).batch(32))

      return aggregated_data

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

    def save_checkpoint(self, round_num):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_round_{round_num}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump((self.state, self.client_learning_rates), f)
        logger.info(f'Checkpoint saved at round {round_num} to {checkpoint_path}')

    def load_checkpoint(self, round_num):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_round_{round_num}.pkl')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                self.state, self.client_learning_rates = pickle.load(f)
            logger.info(f'Checkpoint loaded from round {round_num} from {checkpoint_path}')
        else:
            logger.error(f'Checkpoint file {checkpoint_path} does not exist')

# Generate synthetic data with augmentation
def generate_synthetic_data(num_samples, augment=False):
    X = np.linspace(0, 10, num_samples).astype(np.float32).reshape(-1, 1)
    y = (np.sin(X).ravel() + np.random.normal(0, 0.1, num_samples)).astype(np.float32).reshape(-1, 1)
    if augment:
        X_aug = X + np.random.normal(0, 0.1, X.shape).astype(np.float32)
        y_aug = y + np.random.normal(0, 0.1, y.shape).astype(np.float32)
        X = np.vstack((X, X_aug))
        y = np.vstack((y, y_aug))
    return X, y

# Helper function to create environments and networks for each client
def create_environment_and_network(client_id, variation, config):
    env = gym.make('BipedalWalker-v3')
    env._max_episode_steps = MAX_EPISODES  # Reduce max episode steps to speed up
    env.env.gravity = variation * client_id
    genome = load_genome(BEST_GENOME_PATH)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return env, network, client_id

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
    weight_mutate_power = 1.0
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
    neat_input = tf.keras.Input(shape=(NUM_OUTPUTS,), name='x')  # Ensure this matches the reshaped data
    markov_output = NEATMarkovLayer()(neat_input)
    final_output = tf.keras.layers.Dense(1, activation='sigmoid')(markov_output)

    model = tf.keras.Model(inputs=neat_input, outputs=final_output)
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec={
            'x': tf.TensorSpec(shape=[None, NUM_OUTPUTS], dtype=tf.float32),
            'y': tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
        },
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )

def precompute_neat_outputs(env, network, num_samples):
    states = []
    outputs = []
    for _ in range(num_samples):
        state = env.reset()
        done = False
        while not done:
            action = network.activate(state)  # Get action from NEAT network
            states.append(state)
            outputs.append(action)  # Assuming action needs to be transformed for TensorFlow compatibility
            state, _, done, _ = env.step(action)
    return np.array(states, dtype=np.float32), np.array(outputs, dtype=np.float32)

# Adjust data collection to use precomputed NEAT outputs
def collect_client_data(environment, net, learning_rate, episodes=EPISODES_PER_EVALUATION):
    states, actions = precompute_neat_outputs(environment, net, episodes)
    actions = actions.reshape(-1, NUM_OUTPUTS)  # Make sure actions shape matches NUM_OUTPUTS

    dataset = tf.data.Dataset.from_tensor_slices({
        'x': actions,
        'y': actions[:, 0:1]  # Assuming your model predicts something based on actions
    }).batch(32)
    return dataset

if not os.path.exists(DEMO_FILE):
    env_demo = gym.make('BipedalWalker-v3')
    demos = []
    for _ in range(5):  # Reduce number of demonstrations to speed up
        state = env_demo.reset()
        done = False
        while not done:
            action = env_demo.action_space.sample()  # Random actions as placeholders
            next_state, _, done, _ = env_demo.step(action)
            demos.append((state, [action]))  # Ensure action is wrapped in a list
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
    train_data = [collect_client_data(client[0], client[1], 0.1) for client in clients if collect_client_data(client[0], client[1], 0.1) is not None]
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
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    )
    state = trainer.initialize()

    test = FederatedLearningTest(clients, model_fn, trainer, state, config, demonstrations)
    state, federated_metrics = test.run_federated_training(NUM_ROUNDS)

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