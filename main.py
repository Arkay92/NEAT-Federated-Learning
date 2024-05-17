import neat, gym, pickle, os
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define a custom Keras layer for NEAT network
class NEATLayer(tf.keras.layers.Layer):
    def __init__(self, neat_network, **kwargs):
        super(NEATLayer, self).__init__(**kwargs)
        self.neat_network = neat_network

    def call(self, inputs):
        outputs = tf.vectorized_map(lambda x: tf.convert_to_tensor(self.neat_network.activate(x.numpy().tolist())), inputs)
        return outputs

# Helper function to create environments and networks
def create_environment_and_network(client_id, variation, config):
    env = gym.make('BipedalWalker-v3')
    env.env.gravity = variation * client_id
    genome = pickle.load(open('best_genome.pkl', 'rb'))
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return env, network

# Load NEAT configuration
config_path = './neat_config.txt'
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
    dataset = tf.data.Dataset.from_tensor_slices((np.array(states), np.array(actions)))
    return dataset.batch(32)  # Batch the dataset for federated learning

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

# Testing and Benchmarking Class
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
            self.state, metrics = self.trainer.next(self.state, client_data)
            metrics_list.append(metrics)
            print(f'Round {round_num} metrics:', metrics)
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

        print(f'Average reward after federated learning: {avg_reward}')
        print(f'Baseline reward: {baseline_reward}')
        
        plt.figure(figsize=(10, 5))
        plt.bar(['Baseline', 'Federated Learning'], [baseline_reward, avg_reward])
        plt.ylabel('Average Reward')
        plt.title('Benchmarking')
        plt.show()

# Initialize environments and networks for clients
clients = [create_environment_and_network(i, 1.0 + 0.1 * i, config) for i in range(5)]

# Federated Learning Trainer Setup
trainer = tff.learning.build_federated_averaging_process(model_fn)
state = trainer.initialize()

# Testing and benchmarking
test = FederatedLearningTest(clients, model_fn, trainer, state, config, demonstrations)

# Run federated training
metrics = test.run_federated_training(rounds=10)

# Plot training metrics
test.plot_metrics(metrics)

# Evaluate and plot rewards
neat_network = neat.nn.FeedForwardNetwork.create(pickle.load(open(best_genome_path, 'rb')), config)
rewards = [test.evaluate_model(client[0], neat_network) for client in clients]
test.plot_rewards(rewards)

# Benchmark the model
baseline_reward = 100  # Replace with actual baseline reward for comparison
test.benchmark(baseline_reward)
