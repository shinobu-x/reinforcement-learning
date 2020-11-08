import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, sigma, dt = 1e-2, theta = 0.15, x_init = None):
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x_init = x_init
        self.reset()

    def reset(self):
        self.x_prev = self.x_init if self.x_init is not None \
                else np.zeros_like(self.mu)

    def generate(self, x):
        return self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * \
                np.random.normal(size = self.mu.shape)

    def add_noise(self):
        x = self.x_prev
        dx = self.generate(x)
        self.x_prev = x + dx
        return self.x_prev

    def __repr__(self):
        return 'OrnsteinUhlenbeckNoise(mu = {}, sigma = {}'.format(
                self.mu, self.sigma)
