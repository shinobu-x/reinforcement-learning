import collections
import cv2
import gym
import numpy as np

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        state, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        state, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return state

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env = None, skip = 4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._state_buffer = collections.deque(maxlen = 2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            state, reward, done, info = self.env.step(action)
            self._state_buffer.append(state)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._state_buffer), axis = 0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._state_buffer.clear()
        state = self.env.reset()
        self._state_buffer.append(state)
        return state


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0, high = 255,
                shape = (84, 84, 1), dtype = np.uint8)

    def observation(self, state):
        return ProcessFrame84.process(state)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation =
                cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.state_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                old_space.high.repeat(n_steps, axis = 0), dtype = dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low,
                dtype = self.dtype)
        return self.observation(self.env.reset())

    def observation(self, state):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = state
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0,
                shape = (old_shape[-1], old_shape[0], old_shape[1]),
                dtype = np.float32)

    def observation(self, state):
        return np.moveaxis(state, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, state):
        return np.array(state).astype(np.float32) / 255.0

class MakeEnv(object):
    def __init__(self, env_name):
        super(MakeEnv, self).__init__()
        self.env = gym.make(env_name)
        self.env = MaxAndSkipEnv(self.env)
        self.env = FireResetEnv(self.env)
        self.env = ProcessFrame84(self.env)
        self.env = ImageToPyTorch(self.env)
        self.env = BufferWrapper(self.env, 4)
        self.env = ScaledFloatFrame(self.env)
        self.is_continous = type(self.env.action_space) is gym.spaces.box.Box
        if self.is_continous:
            self.action_space = self.env.action_space.shape[0]
        else:
            self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]

    def init(self):
        return self.env

def get_env_space(env_name):
    env = gym.make(env_name)
    is_continous = type(env.action_space) is gym.spaces.box.Box
    if is_continous:
        action_space = env.action_space.shape[0]
    else:
        action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    return state_space, action_space, is_continous

def multi_agent_make_env(scenario_name, benchmark = False):
    from multiagent import scenarios
    from multiagent.environment import MultiAgentEnv
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    world = scenarios.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                scenario.observation)
    return env
