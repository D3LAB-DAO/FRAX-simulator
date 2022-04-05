class Env:
    def __init__(
        self,
        n_agents,
        n_actions
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions

    def step(self, actions):  # multiple agents' actions
        pass
        # return observe, reward, done, info  # observe => state

    def reset():
        pass


if __name__ == "__main__":

    pass

    # episode = 100
    # step = 1

    # for e in range(episode):
    #     actions = list()
    #     for agent in agents:
    #         actions.append(agent.get_action())

    #     for s in range(step):  # or, while not done:
    #         pass
