r"""This file contain you main logic for accoring to use cases for training and testing.

Organize your code as follows;
    -   If user inputs phase == train, call train function, train the model 
        and save trained weights.
    -   If user inputs phase == test, call test function, load the trained 
        model and note testing accuracy.
    -   Make a use of argparse for command line arguments.
    -   You can also use typing framework to validate the type of a arguments, 
        but it is optional.
    -   Ask if you have some doubts. 
"""
import os
import torch
import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=str)
    parser.add_argument("--model_name", type=str, default="resnet50", help="Name of the model.")
    parser.add_argument("--dataset", type=str, default="vehicle", help="Name of the dataset.")
    parser.add_argument("--data_dir", type=str, default="./datasets/vehicle", help="Path to dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Name of the model.")
    parser.add_argument("--weigths_path", type=str, default="./checkpoints", help="path to trained weights.")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for training.")
    parser.add_argument("--colab", type=bool, default=False, help="Which platform is using for training.")
    parser.add_argument("--out_path", type=str, default="./outputs", help="Save training metrics here.")
    parser.add_argument("--momentum", type=int, default=0.09, help="path to trained weights.")
    parser.add_argument("--num_epochs", type=int, default=10, help="for # iteration model will run on data.")
    parser.add_argument("--num_classes", type=float, default=2, help="numuber of objects + background.")
    parser.add_argument("--best_acc", type=float, default=0, help="undertanding the value of accuracy.")
    parser.add_argument("--print_itr", type=int, default=1, help="Print after # iterations.")
    parser.add_argument("--num_workers", type=int, default=1, help="num of workers")
    
    args = parser.parse_args()
    # print(args.model_name)
    os.makedirs(args.weigths_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, args.model_name), exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"Training on {device}.")

    if args.phase == "train":
        import train
        print(f"Training the model {args.model_name}")
        train.run(args, device)
    elif args.phase == "test":
        import test
        test.run(args, device)
    else:
        raise ValueError("You have entered invalid value.")

