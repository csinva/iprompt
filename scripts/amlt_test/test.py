import pickle as pkl
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--n', type=int, default=2,
                    help='an integer for the accumulator')

args = parser.parse_args()

print('printing hello!', args.n)

pkl.dump({'hello': 'world' + str(args.n)}, open('test.pkl', 'w'))
