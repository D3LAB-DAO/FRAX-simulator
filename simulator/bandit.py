import numpy as np


def soften(
    x,  # np.array
    t=2.0  # temperature
):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x = x / t  # temperature scaling
    e_x = np.exp(x - np.max(x))  # prevent overflow
    return e_x / np.sum(e_x)


class SoftmaxBandit:
    """
    MAB (Multi Armed Bandit)
    - No state
    - Only action and reward are used
    """

    def __init__(
        self,
        # state_size,
        action_size
    ):
        # self.state_size = state_size
        self.actions = list(range(action_size))  # [0, 1, 2, ...]
        self.last_action = 0  # index of the most recent action
        self.q_table = np.zeros_like(self.actions)

    def get_action(self, deterministic=False):  # , state=None):
        """
        Using a Boltzmann distribution
        - Same as Softmax + Temperature
        - Can do annealing
        """
        if deterministic:
            action = np.argmax(self.actions)
            check = np.where(self.actions == self.actions[action])[0]
            if len(check) != 1:
                action = np.random.choice(check)
        else:
            action = np.random.choice(self.actions, 1, p=soften(self.q_table))

        self.last_action = action = int(action)
        return action

    def learn(self, action, reward, lr=0.01):
        """
        No action counter
        - Can do online learning (accept entering of new user)
        """
        q1 = self.q_table[action]
        q2 = reward
        g = 1 / soften(self.q_table)[action]
        self.q_table[action] += lr * g * (q2 - q1)

    def reset(self):
        self.last_action = 0
        self.q_table = np.zeros_like(self.actions)
