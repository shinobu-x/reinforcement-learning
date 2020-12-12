import numpy as np

class GridWorld(object):
    def __init__():
        self.satte = np.zeros((4, 4, 4))
        self.player = np.array([0, 0, 0, 1])
        self.wall = np.array([0, 0, 1, 0])
        self.pit = np.array([0, 1, 0, 0])
        self.goal = np.array([1, 0, 0, 0])

    def find(state, obj):
        for i in range(0, 4):
            for j in range(0, 4):
                if (state[i, j] == obj).all(): return i, j

    def init_grid():
        state = self.state
        state[0, 1] = self.player
        state[2, 2] = self.wall
        state[1, 1] = self.pit
        state[3, 3] = self.goal
        return state

    def init_player(random = False):
        state = self.state
        state[(np.random.randint(0, 4), np.random.randint(0, 4))] = self.player
        if random:
            state[(np.random.randint(0, 4), np.random.randint(0, 4))] = \
                    self.wall
            state[(np.random.randint(0, 4), np.random.randint(0, 4))] = \
                    self.pit
            state[(np.random.randint(0, 4), np.random.randint(0, 4))] = \
                    self.goal
        else:
            state[2, 2] = self.wall
            state[1, 1] = self.pit
            state[1, 2] = self.goal
        player = find(state, self.player)
        wall = find(state, self.wall)
        pit = find(state, self.pit)
        goal = find(state, self.goal)
        return init_player() \
                if (not player or not wall or not pit or not goal) else state

    def move(state, action):
        player = find(state, self.player)
        wall = find(state, self.wall)
        pit = find(state, self.pit)
        goal = find(state, self.goal)
        state = self.state
        actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        pos = (player[0] + actions[action][0], player[1] + actions[action][1])
        if pos != wall:
            if (np.array(pos) <= (3, 3)).all() and \
                    (np.array(pos) >= (0, 0)).all():
                state[pos][3] = 1
        if not find(state, self.player):
            state[player] = self.player
        state[wall][2] = 1
        state[pit][1] = 1
        state[goal][0] = 1
        return state

    def get_pos(state, index):
        for i in range(0, 4):
            for j in range(0, 4):
                if state[i, j][index] == 1: return i, j

    def get_reward(state):
        player = get_pos(state, 3)
        pit = get_pos(state, 1)
        goal = get_pos(state, 0)
        if player == pit: return -10
        elif player == goal: return 10
        else: return -1

    def display_state(state):
        grid = np.zeros((4, 4), dtype = str)
        player = find(state, self.player)
        wall = find(state, self.wall)
        pit = find(state, self.pit)
        goal = find(state, self.goal)
        for i in range(0, 4):
            for j in range(0, 4):
                grid[i, j] = ''
        if player: gird[player] = 'P'
        if wall: grid[wall] = 'W'
        if pit: grid[pit] = '-'
        if goal: grid[goal] = '+'
        return grid
