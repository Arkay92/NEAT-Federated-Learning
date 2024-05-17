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

# Load NEAT configuration and initialize population
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

population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
winner = population.run(lambda genomes, config: evaluate_genomes(genomes, config, create_environment_and_network(1, 1.0, config)[0]), 30)

# Save the best genome for initialization
with open('best_genome.pkl', 'wb') as f:
    pickle.dump(winner, f)

# Define TensorFlow Federated model function using the NEAT genome
def model_fn():
    # Load and convert the best genome to a NEAT network
    genome = pickle.load(open('best_genome.pkl', 'rb'))
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

# Simulate federated learning across clients
clients = [create_environment_and_network(i, 1.0 + 0.1 * i, config) for i in range(5)]
trainer = tff.learning.build_federated_averaging_process(model_fn)
state = trainer.initialize()

for round_num in range(10):
    client_data = [collect_client_data(client[0], client[1]) for client in clients]
    state, metrics = trainer.next(state, client_data)
    print(f'Round {round_num} metrics:', metrics)
