import gym
import numpy as np
import pybullet_envs
env = gym.make('MinitaurBulletEnv-v0')
def init(env, q = False):
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    actions = np.zeros(action_space)
    state_values = np.zeros(state_space)
    action_values = np.zeros(state_space + action_space)
    eligibilities = np.zeros(state_space) if q is False \
            else np.zeros(state_space + action_space)
    state = env.reset()
    action = [0.0] * 8
    action = [0.0 - 0.1 * (-1 if i % 2 == 0 else 1) * (-1 if i < 4 else 1)
            for i in range(8)]
    return state_values, action_values, eligibilities, state, action
LAMBDA = 0.95
GAMMA = 0.01
ALPHA = 0.01
state_values, _, eligibilities, state, action = init(env)
for t in range(20):
    next_state, reward, _, _ = env.step(action)
    eligibilities *= LAMBDA * GAMMA
    # e(s) = e(s) + 1
    eligibilities[state.astype(int)] += 1.0
    # \delta = r + \gamma * \mathcal{V}(s^{\prime}) - \mathcal{V}(s)
    delta = reward + GAMMA * state_values[next_state.astype(int)] - \
            state_values[state.astype(int)]
    state_values = state_values + ALPHA * delta * eligibilities
    # e(s) = \gamma * \lambda * e(s)
    eligibilities[state.astype(int)] *= GAMMA * LAMBDA
_, action_values, eligibilities, state, action = init(env, q = True)
for t in range(20):
    next_state, reward, _, _ = env.step(action)
    next_action = action
    eligibilities *= LAMBDA * GAMMA
    # e(s,a) = e(s,a) + 1
    eligibilities[np.concatenate((state.astype(int),
        np.array(action).astype(int)))] += 1.0
    # \delta = r + \gamma * \mathcal{Q}(s^{prime},a^{prime}) - \matchcal{Q}(s,a)
    delta = reward + GAMMA * \
            action_values[np.concatenate((next_state.astype(int),
                np.array(next_action).astype(int)))] - \
            action_values[np.concatenate((state.astype(int),
                np.array(action).astype(int)))]
    action_values = action_values + ALPHA * delta * eligibilities
    # e(s,a) = \gamma * \lambda * e(s,a)
    eligibilities[np.concatenate((state.astype(int),
        np.array(action).astype(int)))] *= LAMBDA * GAMMA
