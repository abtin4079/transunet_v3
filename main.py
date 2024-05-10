import sys
import torch
import argparse

# Additional Scripts
from train import TrainTestPipe
from inference import SegInference
import torch.nn as nn
from torch.nn.parallel import DataParallel


def main_pipeline(parser):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s) available.")
        if num_gpus > 1:
            device = 'cuda'
            print("Multiple GPUs available. Using", device)
        else:
            device = 'cuda:0'  # or 'cuda:0' if you want to use the first GPU
            print("Single GPU available. Using", device)
    else:
        device = 'cpu'
        print("No GPU available. Using CPU.")

    if parser.mode == 'train':
        ttp = TrainTestPipe(train_path=parser.train_path,
                            train_sail_path=parser.train_sail_path,
                            test_path=parser.test_path,
                            test_sail_path=parser.test_sail_path,
                            model_path=parser.model_path,
                            device=device)
        
        ttp.setup_model()

        if num_gpus > 1:
            # Wrap the model with DataParallel
            ttp.model = DataParallel(ttp.model)


        ttp.train()

    elif parser.mode == 'inference':
        inf = SegInference(model_path=parser.model_path,
                           device=device)

        if num_gpus > 1:
            # Wrap the model with DataParallel
            inf.setup_model()
            inf.model = DataParallel(inf.model)

        _ = inf.infer(parser.image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'])
    parser.add_argument('--model_path', required=True, type=str, default=None)

    parser.add_argument('--train_path', required='train' in sys.argv,  type=str, default=None)
    parser.add_argument('--train_sail_path', required='train' in sys.argv,  type=str, default=None)
    parser.add_argument('--test_path', required='train' in sys.argv, type=str, default=None)
    parser.add_argument('--test_sail_path', required='train' in sys.argv, type=str, default=None)

    parser.add_argument('--image_path', required='infer' in sys.argv, type=str, default=None)
    parser = parser.parse_args()

    main_pipeline(parser)
