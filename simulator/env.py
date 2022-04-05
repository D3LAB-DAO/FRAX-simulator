from agent import Agent


class Env:
    def __init__(
        self,
        n_agents,
        n_actions
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions

    def step(self, action):
        # return observe, reward, done, info
        # `observe`` is same as `state`
        if action == 1:
            # enough reward
            # consider: lr=0.01 as default
            return None, 32, None, None
        else:
            return None, -100, None, None

    def reset(self):
        pass


if __name__ == "__main__":
    N_AGENTS = 5
    N_ACTIONS = 4
    N_EPISODE = 1  # experiments
    N_STEPS = 1000  # trials

    agents = [Agent(N_ACTIONS) for _ in range(N_AGENTS)]
    env = Env(N_AGENTS, N_ACTIONS)

    for e in range(N_EPISODE):
        env.reset()

        for s in range(N_STEPS):  # or, while not done:
            # actions = list()  # log

            for agent in agents:
                action = agent.get_action()
                # actions.append(action)  # log

                _, reward, _, _ = env.step(action)
                agent.learn(action, reward)

            # print(actions, agents[0].q_table)  # log

        # log
        deterministic_actions = [agent.get_action(deterministic=True) for agent in agents]
        print(">>>", deterministic_actions)
