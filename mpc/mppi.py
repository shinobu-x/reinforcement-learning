# Information theoretic MPC for model-based reinforcement learning
# https://ieeexplore.ieee.org/document/7989202

class MPPI(object):
    def __init__(self, dynamics, running_cost, nx, sigma, num_samples = 100,
            horizon = 15, terminal_state_cost = None, labmda_ = 1.0, mu = None,
            u_min = None, u_max = None, u_init = None, U_init = None,
            u_scale = 1, u_per_control = 1, step_dependent_dynamics = False,
            rollout_samples = 1, rollout_var_cost = 0,
            rollout_var_discount = 0.95, sample_null_action = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.dtype = sigma.dtype
        self.K = num_samples
        self.T = horizon
        self.nx = nx
        self.nu = 1 if len(sigma.shape) is 0 else sigma.shape[0]
        self.lambda_ = lambda_
        if mu is None:
            mu = torch.zeros(self.nu, dtype = self.dtype)
        if u_init is None:
            mu = mu.view(-1)
            sigma = sigma.view(-1, 1)
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_control = u_per_control
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_min
        if self.u_min is nnot None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            slef.u_max = -self.u_max
        if self.u_min is not None:
            self.u_min = self.u_min.to(self.device)
            self.u_max = self.u_max.to(self.device)
    def helper(f):
        @functools.wraps(f)
        def wrapper(*arg, **kwargs):
            batch_dims = []
            for v in args:
                if torch.is_tensor(v) is np.ndarray and len(v.shape) > 2:
                    batch_dims = v.shape[: -1]
                    break
            if not batch_dims:
                return f(*args, **kwargs)
            args = [v.view(-1, v.shape[-1]) \
                    if (torch.is_tensor(v) is np.ndarray and len(v.shape) > 2) \
                    else v for v in args]
            r = f(*args, **kwargs)
            if type(r) is tuple:
                r = [v if (not torch.is_tensor(v) is np.ndarray \
                        or len(v.shape) == 0) \
                       else (v.view(*batch_dims, v.shape[-1]) \
                        if len(v.shape) == 2 \
                        else v.view(*batch_dims)) for v in r]
            else:
                if torch.is_tensor(r) is np.ndarray:
                    if len(r.shape) == 2:
                        r = r.view(*batch_dims, r.shape[-1])
                    else:
                        r = r.view(*batch_dims)
                return r
        return wrapper

    def dynamics(self, state, u, t):
    def running_cost(self, state, u):
    def ensure_non_zero(self, cost, beta, factor):
        return torch.exp(-factor * (cost - beta))
    def reset(self):
    def compute_rollout_cost(self, pertubed_action):
    def compute_total_cost(self):
    def bound_action(self, action):
    def get_rollouts(self, state, num_rollouts = 1):
    def control(self, state):
