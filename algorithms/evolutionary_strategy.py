import multiprocessing as mp
import numpy as np

class EvolutionaryStrategy(object):
    def __init__(self, wieghts, get_rewards, num_population = 50, sigma = 1e-1,
            lr = 1e-2, lr_decay = 0.9, num_threds = 1):
        self.weights = weights
        self.get_rewards = get_rewards
        self.num_population = num_population
        self.sigma = sigma
        self.lr = lr
        self.decay = decay
        self.num_threads = num_threads

    def get_weights(self, weight = None, population = None):
        if population is None:
            return self.weights
        else:
            weights = []
            for index, i in enumerate(population):
                perturbed = self.sigma * i
                weights.append(weight[index] + perturbed)
            return weights

    def get_population(self):
        population = []
        for i in range(self.num_population):
            noise = []
            for weight in self.weights:
                noise.append(np.random.randn(*weight.shape))
            population.append(noise)
        return population

    def get_all_rewards(self, thread_pool, population):
        if pool is not None:
            rewards = [self.get_rewards(self.get_weights(self.weight, p))
                    for p in population]
        else:
            rewards = []
            for p in population:
                weights = self.get_weights(self.weight, p)
                rewards.append(self.get_reward(weights))
        return np.array(rewards)

    def update_weights(self, rewards, population):
        try:
            rewards = (rewards - rewards.mean()) / rewards.std()
        except ZeroDivisionError:
            return
        for index, weight in enumerate(self.weights):
            populationt = np.array([p[index] for p in population])
            delta = self.lr / (self.num_population * self.sigma)
            self.weights[index] = w + delta * np.dot(population.T.rewards).T
            self.lr = self.lr_decay

    def run(self, iterations, frequency = 10):
        thread_pool = mp.Pool(self.num_threads) if self.num_threads > 1 \
                else None
        for iteratin in range(iteratins):
            population = self.get_population()
            rewards = self.get_all_rewards(thread_pool, population)
            self.update_weights(rewards, population)
            i = iteration + 1
            if i % frequency == 0:
                print('Iteration %d. Reward: %f' % (i, self.get_rewards(
                    self.weights)))
        if pool is not None:
            thread_pool.close()
            thread_pool.join()
