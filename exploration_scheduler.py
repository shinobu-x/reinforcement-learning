class ExplorationScheduler(object):
    def __init__(self, schedule_timesteps, final_probability,
            initial_probability = 1.0):
        self.schedule_timesteps = schedule_timesteps
        self.initial_probability = initial_probability
        self.final_probability = final_probability

    def value(self, timestep):
        return self.final_probability + \
                min(float(timestep) / self.schedule_timesteps, 1.0) * \
                (self.final_probability - self.initial_probability)
