import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from vit_model import ViT
from torch.utils.data import DataLoader
import pandas as pd
from grid import make_grid


class GridDataset(Dataset):
    def __init__(self, complex_dir, ):
        self.complex_dir = complex_dir
        self.complex_grids = list(os.listdir(complex_dir))       

    def __len__(self):
        return len(self.complex_grids)


    def __getitem__(self, idx):
        
        file_name = self.complex_grids[idx]
        grid_path = os.path.join(self.complex_dir, self.complex_grids[idx])
        positions, features = self.get_grid_data(grid_path)

        grid = make_grid(positions, features, 1.0, 10)
        grid = torch.from_numpy(grid.copy())

        return np.float32(grid), file_name

    def get_grid_data(self, grid_data_path):
        grid_data = np.load(grid_data_path)
        positions, features = grid_data["positions"], grid_data["features"]
        return positions, features


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT model on grid dataset")

    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--grid_dir", type=str, required=True, help="Directory containing grid features")
    parser.add_argument("--output_file", type=str, required=True, help="path to output csv file")


    parser.add_argument("--depth", type=int, default=1, help="Number of transformer blocks")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--mlp_dim", type=int, default=512, help="MLP hidden layer dimension")
    parser.add_argument("--dim_head", type=int, default=128, help="Dimension per attention head")
    parser.add_argument("--channels", type=int, default=33, help="Number of input channels")
    parser.add_argument("--patch_size", nargs=3, type=int, default=(3, 3, 3), help="Patch size (x y z)")
    parser.add_argument("--grid_size", nargs=3, type=int, default=(21, 21, 21), help="Grid size (x y z)")


    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    data = GridDataset(args.grid_dir)
    dataloader = DataLoader(data, batch_size=64, shuffle=False)


    model = ViT(args.grid_size, args.patch_size, depth=args.depth, heads=args.heads, 
                dim=args.dim, mlp_dim=args.mlp_dim, channels=args.channels)


    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()

    predictions = []
    ids = []

    with torch.no_grad():
        for eval_data, eval_labels in dataloader:
            target = model(eval_data.to(device)).flatten().detach().cpu().numpy()
            predictions.append(target)
            ids.extend(list(eval_labels))

   
    predictions = np.concatenate(predictions)

    data = pd.DataFrame({"ids": ids, "predictions": predictions})

    data.to_csv(args.output_file, index=False)
