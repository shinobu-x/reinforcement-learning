import numpy as np
from copy import deepcopy

# A0C: Alpha Zero in Continuous Action Space
# https://arxiv.org/abs/1805.09613
class State():
    def __init__(self, state, reward, done, parent_action, action_space,
            model):
        self.state = state
        self.reward = reward
        self.parent_action = parent_action
        self.count = 0
        self.model = model
        self.evaluate()
        self.action_space = action_space
        self.child_actions = [Action(action, parent_state = self,
            q_value = self.value) for action in range(action_space)]
        self.priors = model.actor(state)

    def select(self, coefficient = 1.5):
        # \pi_{tree}(a | s) = argmax_a(Q(s, a) + c_{puct} * \pi_{\phi}(a | s) *
        # \sqrt(n(s) / (n(s, a) + 1))
        uct = np.array([child_action.q_value + prior * coefficient *
            (np.sqrt(self.count + 1) / (child_action.count + 1)) \
                    for child_action, prior \
                    in zip(self.child_actions, self.priors)])
        action = np.argmax(uct)
        return self.child_actions[action]

    def evaluate(self):
        self.value = np.squeeze(self.model.critic(self.state)) \
                if not self.done else np.array(0, 0)

    def update(self):
        self.count += 1

class Action():
    def __init__(self, action, parent_state, q_value = 0.0):
        self.action = action
        self.parent_state = parent_state
        self.w = 0.0
        self.count = 0
        self.q_value = q_value

    def add_child_state(self, next_state, reward, done, model):
        self.child_state = State(next_state, done,
                self.parent_state.action_space, model)

    def update(self, reward):
        self.count += 1
        self.reward += reward
        self.q_value = self.rewards / self.count

class MCTS():
    def __init__(self, root, root_index, model, action_space, gamma):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.action_space = action_space
        self.gamma = gamma

    def search(self, mcts_iters, coefficient, env, mcts_env):
        if self.root is None:
            self.root = State(self.root_index, reward = 0.0, done = False,
                    parent_action = None, action_space = self.action_space,
                    model = self.model)
        else:
            self.root.parent_action = None
        if self.root.done:
            return
        for i in range(mcts_iters):
            state = self.root
            mcts_env = deepcopy(env)
        while not state.done:
            action = state.select(coefficient = coefficient)
            next_state, reward, done, _ = mcts_env.step(action.action)
            if hasattr(action, 'child_state'):
                state = action.child_state
                continue
            else:
                state = action.add_child_state(next_state, reward, done,
                        self.model)
                break
        reward = state.value
        while state.parent_action is not None:
            # R(s_i, a_i) = r(s_i, a_i) + \gamma * R(s_{i+1}, a_{i+1})
            reward = state.reward + self.gamma * reward
            action = state.parent_action
            action.update(reward)
            state = action.parent_state
            state.update()

    def compute_results(self, tau = 20):
        counts = np.array([child_action.count for child_action
            in self.root.child_actions])
        q_value = np.array([child_action.q_value for child_action
            in self.root.child_actions])
        # \hat{\pi}(a_i | s) = n(s, a_i)^{\tau} / Z(s, \tau)
        x = (counts / np.max(counts)) ** tau
        policy_target = np.abs(x / np.sum(x))
        value_target = np.sum((counts / np.sum(counts)) * q_value)[None]
        return self.root.state, policy_target, value_target

    def forward(self, action, next_state):
        if not hasattr(self.root.child_actions[action], 'child_state') or \
                np.linarg.norm(
                        self.root.child_actions[action].child_state.state -
                        next_state) > 1e-2:
            self.root = None
            self.root_index = next_state
        else:
            self.root = self.root.child_actions[action].child_state
