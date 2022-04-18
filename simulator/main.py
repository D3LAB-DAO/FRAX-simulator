from args import parser

from agent.DRL.DQN import DQNAgent
# from agent.DRL.Double import DDQNAgent
# from agent.DRL.Dueling import DuelingAgent, D3QNAgent
    
if __name__ == "__main__":
    args = parser()
    print(args)

    # N_AGENTS = 5
    # N_ACTIONS = 4
    # N_EPISODE = 1  # experiments
    # N_STEPS = 1000  # trials

    # agents = [Agent(N_ACTIONS) for _ in range(N_AGENTS)]
    # env = Env(N_AGENTS, N_ACTIONS)

    # for e in range(N_EPISODE):
    #     env.reset()

    #     for s in range(N_STEPS):  # or, while not done:
    #         # actions = list()  # log

    #         for agent in agents:
    #             action = agent.get_action()
    #             # actions.append(action)  # log

    #             _, reward, _, _ = env.step(action)
    #             agent.learn(action, reward)

    #         # print(actions, agents[0].q_table)  # log

    #     # log
    #     deterministic_actions = [agent.get_action(deterministic=True) for agent in agents]
    #     print(">>>", deterministic_actions)
