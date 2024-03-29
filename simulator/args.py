import argparse


def argparser():
    parser = argparse.ArgumentParser(description='Hyperparameters')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    parser.add_argument('--episodes', metavar='E', type=int, default=50000,
                        help='Total episodes')
    parser.add_argument('--double', metavar='D', type=bool, default=False,
                        help='Enable Double DQN')
    parser.add_argument('--dueling', metavar='B', type=bool, default=False,
                        help='Enable Dueling DQN')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    print(args)
