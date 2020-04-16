import torch
from parser import build_parser
from data import get_data, get_dataset, split_data, get_loaders
from model import get_model
from trainer import get_trainer


def main(args):
    x, fx = get_data(args)

    device = torch.device("cuda" if args.cuda else "cpu")
    dataset = get_dataset(x, fx)
    train_idx, val_idx = split_data(args.split)

    train_loader, val_loader = get_loaders(dataset, train_idx, val_idx)
    
    model = get_model(args)

    max_epochs = 50



if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
