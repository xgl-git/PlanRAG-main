import argparse


def init_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./dataset', type=str)
    parser.add_argument('--storage', default='./storage', type=str)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--device', default='', type=str)
    parser.add_argument('--retriever', default='bm25', type=str)
    parser.add_argument('--api_key', default="", type=str)
    parser.add_argument('--sinkhorn', action='store_true')
    parser.add_argument('--epsilon', default=0.1, type = float)
    parser.add_argument('--stopThr', default=1e-6, type = float)
    parser.add_argument('--numItermax', default=1000, type=int)
    parser.add_argument('--dim', default=384, type=int)
    return parser
