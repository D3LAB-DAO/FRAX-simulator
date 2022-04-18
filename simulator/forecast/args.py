import argparse


def argparser():
    parser = argparse.ArgumentParser(description='Hyperparameters')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    parser.add_argument('--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('--reset', action='store_true',
                        help='Enable Force Reset')
    parser.add_argument('--vals', metavar='V', type=int, default=21,
                        help='Number of val dataset')
    parser.add_argument('--preds', metavar='P', type=int, default=360,
                        help='Number of predict dataset')
     
    parser.add_argument('--seed', metavar='S', type=int, default=950327,
                        help='Random seed')
    
    parser.add_argument('--load', action='store_true',
                        help='Load model')
    parser.add_argument('--best', action='store_true',
                        help='Load best/recent model')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate model')
    parser.add_argument('--test', action='store_true',
                        help='Backtest model')
    parser.add_argument('--pred', action='store_true',
                        help='Predict/Inference model')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    print(args)
