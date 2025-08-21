import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from vit_model import ViT
import pandas as pd
from grid import make_grid
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr


class GridDataset(Dataset):
    def __init__(self, annotations_file, complex_dir):
        self.complex_anno = pd.read_csv(annotations_file)
        self.complex_dir = complex_dir
       

    def __len__(self):
        return len(self.complex_anno)


    def __getitem__(self, idx):
        grid_path = os.path.join(self.complex_dir, f"{self.complex_anno.iloc[idx, 0]}.npy.npz")
        positions, features = self.get_grid_data(grid_path)
        label = self.complex_anno.iloc[idx, 1]
        
        grid = make_grid(positions, features, 1.0, 10)
        grid = torch.from_numpy(grid.copy())

        return np.float32(grid), np.float32(label)

    def get_grid_data(self, grid_data_path):
        grid_data = np.load(grid_data_path)
        positions, features = grid_data["positions"], grid_data["features"]
        return positions, features


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT model on grid dataset")
    
    parser.add_argument("--grid_dir", type=str, required=True, help="Directory containing grid features")
    parser.add_argument("--coreset_2016", type=str, required=True, help="path containing coreset 2016 affinity data")
    parser.add_argument("--coreset_2013", type=str, required=True, help="path containing coreset 2013 affinity data")

    parser.add_argument("--model",  type=str, required=True, help="pth file model")
    
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

    CASF_data = GridDataset(args.coreset_2016, args.grid_dir)
    CASF_dataloader = DataLoader(CASF_data, batch_size=64, shuffle=False)

    pdbbind_2013_core = GridDataset(args.coreset_2013, args.grid_dir)
    pdbbind_2013_core_dataloader = DataLoader(pdbbind_2013_core, batch_size=64, shuffle=False)

    model =ViT(args.grid_size, args.patch_size, depth=args.depth, heads=args.heads, 
               dim=args.dim, mlp_dim=args.mlp_dim, channels=args.channels)

    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()


    CASF_pred = []
    CASF_true = []

    pdbbind_2013_core_pred = []
    pdbbind_2013_core_true = []

    with torch.no_grad():
        for eval_data, eval_labels in CASF_dataloader:
            target = model(eval_data.to(device)).flatten().detach().cpu().numpy()
            CASF_pred.append(target)
            CASF_true.append(eval_labels.flatten().detach().numpy())

        for eval_data, eval_labels in pdbbind_2013_core_dataloader:
            target = model(eval_data.to(device)).flatten().detach().cpu().numpy()
            pdbbind_2013_core_pred.append(target)
            pdbbind_2013_core_true.append(eval_labels.flatten().detach().numpy())
   
    predictions = np.concatenate(pdbbind_2013_core_pred)
    true_labels = np.concatenate(pdbbind_2013_core_true)
    mae_pdbbibd_2013 = mean_absolute_error(true_labels, predictions)
    pcc_pdbbibd_2013 = pearsonr(true_labels, predictions)[0]
    rmse_pdbbind_2013 = root_mean_squared_error(true_labels, predictions)


    predictions = np.concatenate(CASF_pred)
    true_labels = np.concatenate(CASF_true)

    mae_CASF_2016 = mean_absolute_error(true_labels, predictions)
    pcc_CASF_2016 = pearsonr(true_labels, predictions)[0]
    rmse_CASF_2016 = root_mean_squared_error(true_labels, predictions)


    print(f"CASF mae {mae_CASF_2016} pcc {pcc_CASF_2016} rmse {rmse_CASF_2016}")
    print(f"pdbbind 2013 mae {mae_pdbbibd_2013} pcc {pcc_pdbbibd_2013} rmse {rmse_pdbbind_2013}")
