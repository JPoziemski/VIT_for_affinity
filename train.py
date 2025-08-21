import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr
from vit_model import ViT
from grid import rotate, make_grid
import random


torch.manual_seed(0)
random.seed(0)
class GridDataset(Dataset):
    def __init__(self, annotations_file, complex_dir, augmentation=False):
        self.complex_anno = pd.read_csv(annotations_file)
        self.complex_dir = complex_dir

       
        self.augmentation_func = {0: lambda pos: pos,
                                  1: lambda pos: rotate(pos, np.array([1, 0, 0]), 90),
                                  2: lambda pos: rotate(pos, np.array([1, 0, 0]), 180),
                                  3: lambda pos: rotate(pos, np.array([1, 0, 0]), 270),
                                  4: lambda pos: rotate(pos, np.array([0, 1, 0]), 90),
                                  5: lambda pos: rotate(pos, np.array([0, 1, 0]), 180),
                                  6: lambda pos: rotate(pos, np.array([0, 1, 0]), 270),
                                  7: lambda pos: rotate(pos, np.array([0, 0, 1]), 90),
                                  8: lambda pos: rotate(pos, np.array([0, 0, 1]), 180),
                                  9: lambda pos: rotate(pos, np.array([0, 0, 1]), 270),
                                  10: lambda pos: rotate(pos, np.array([1, 1, 0]), 90),
                                  11: lambda pos: rotate(pos, np.array([1, 1, 0]), 180),
                                  12: lambda pos: rotate(pos, np.array([1, 1, 0]), 270),
                                  13: lambda pos: rotate(pos, np.array([0, 1, 1]), 90),
                                  14: lambda pos: rotate(pos, np.array([0, 1, 1]), 180),
                                  15: lambda pos: rotate(pos, np.array([0, 1, 1]), 270),
                                  16: lambda pos: rotate(pos, np.array([1, 0, 1]), 90),
                                  17: lambda pos: rotate(pos, np.array([1, 0, 1]), 180),
                                  18: lambda pos: rotate(pos, np.array([1, 0, 1]), 270),
                                  19: lambda pos: rotate(pos, np.array([1, -1, -1]), 90),
                                  20: lambda pos: rotate(pos, np.array([1, -1, -1]), 180),
                                  21: lambda pos: rotate(pos, np.array([1, -1, -1]), 270),
                                  22: lambda pos: rotate(pos, np.array([-1, 1, -1]), 90),
                                  23: lambda pos: rotate(pos, np.array([-1, 1, -1]), 180),
                                  24: lambda pos: rotate(pos, np.array([-1, 1, -1]), 270),
                                  25: lambda pos: rotate(pos, np.array([-1, -1, 1]), 90),
                                  26: lambda pos: rotate(pos, np.array([-1, -1, 1]), 180),
                                  27: lambda pos: rotate(pos, np.array([-1, -1, 1]), 270),
                                  28: lambda pos: rotate(pos, np.array([1, 1, -1]), 90),
                                  29: lambda pos: rotate(pos, np.array([1, 1, -1]), 180),
                                  30: lambda pos: rotate(pos, np.array([1, 1, -1]), 270),
                                  31: lambda pos: rotate(pos, np.array([-1, 1, 1]), 90),
                                  32: lambda pos: rotate(pos, np.array([-1, 1, 1]), 180),
                                  33: lambda pos: rotate(pos, np.array([-1, 1, 1]), 270),
                                  34: lambda pos: rotate(pos, np.array([1, -1, 1]), 90),
                                  35: lambda pos: rotate(pos, np.array([1, -1, 1]), 180),
                                  36: lambda pos: rotate(pos, np.array([1, -1, 1]), 270),
                                  37: lambda pos: rotate(pos, np.array([1, 1, 1]), 90),
                                  38: lambda pos: rotate(pos, np.array([1, 1, 1]), 180),
                                  39: lambda pos: rotate(pos, np.array([1, 1, 1]), 270),
                                  40: lambda pos: rotate(pos, np.array([1, 0, -1]), 90),
                                  41: lambda pos: rotate(pos, np.array([1, 0, -1]), 180),
                                  42: lambda pos: rotate(pos, np.array([1, 0, -1]), 270),
                                  43: lambda pos: rotate(pos, np.array([1, -1, 0]), 90),
                                  44: lambda pos: rotate(pos, np.array([1, -1, 0]), 180),
                                  45: lambda pos: rotate(pos, np.array([1, -1, 0]), 270),
                                  46: lambda pos: rotate(pos, np.array([0, 1, -1]), 90),
                                  47: lambda pos: rotate(pos, np.array([0, 1, -1]), 180),
                                  48: lambda pos: rotate(pos, np.array([0, 1, -1]), 270),
                                  49: lambda pos: rotate(pos, np.array([-1, 1, 0]), 90),
                                  50: lambda pos: rotate(pos, np.array([-1, 1, 0]), 180),
                                  51: lambda pos: rotate(pos, np.array([-1, 1, 0]), 270),
                                  52: lambda pos: rotate(pos, np.array([0, -1, 1]), 90),
                                  53: lambda pos: rotate(pos, np.array([0, -1, 1]), 180),
                                  54: lambda pos: rotate(pos, np.array([0, -1, 1]), 270),
                                  55: lambda pos: rotate(pos, np.array([-1, 0, 1]), 90),
                                  56: lambda pos: rotate(pos, np.array([-1, 0, 1]), 180),
                                  57: lambda pos: rotate(pos, np.array([-1, 0, 1]), 270),
                                  58: lambda pos: rotate(pos, np.array([-1, -1, -1]), 90),
                                  59: lambda pos: rotate(pos, np.array([-1, -1, -1]), 180),
                                  60: lambda pos: rotate(pos, np.array([-1, -1, -1]), 270),
                                  61: lambda pos: rotate(pos, np.array([0, -1, -1]), 90),
                                  62: lambda pos: rotate(pos, np.array([0, -1, -1]), 180),
                                  63: lambda pos: rotate(pos, np.array([0, -1, -1]), 270),

                                  }
      


        self.augmentation = augmentation
        if augmentation:
            self.add_augmentation()
    def __len__(self):
        return len(self.complex_anno)

    def add_augmentation(self):
        repeated_data = [self.complex_anno.copy() for _ in range(len(self.augmentation_func))]
        #print(repeated_data)
        for i, data in enumerate(repeated_data):
            data["augmentation"] = i
        #print(data)
        df_repeated = pd.concat(repeated_data, ignore_index=True)
        df_repeated = df_repeated.sample(frac=1.0, random_state=42)
        self.complex_anno = df_repeated

    def remove_features(self, feature_matrix, col_to_delete):
        feature_matrix = np.delete(feature_matrix, col_to_delete, axis=1)
        return feature_matrix

    def __getitem__(self, idx):
        grid_path = os.path.join(self.complex_dir, f"{self.complex_anno.iloc[idx, 0]}.npy.npz")
        positions, features = self.get_grid_data(grid_path)
        label = self.complex_anno.iloc[idx, 1]
        if self.augmentation:
            aug_code = self.complex_anno.iloc[idx, 2]
            aug_transformation = self.augmentation_func[aug_code]
            positions = aug_transformation(positions)

        grid = make_grid(positions, features, 1.0)
        grid = torch.from_numpy(grid.copy())

        return np.float32(grid), np.float32(label)

    def get_grid_data(self, grid_data_path):
        grid_data = np.load(grid_data_path)
        positions, features = grid_data["positions"], grid_data["features"]
        return positions, features


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT model on grid dataset")


    parser.add_argument("--train_data", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--valid_data", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--coreset_2016", type=str, required=True, help="Path to CASF CSV")
    parser.add_argument("--coreset_2013", type=str, required=True, help="Path to PDBbind 2013 core csv")
    parser.add_argument("--grid_dir", type=str, required=True, help="Directory containing grid features")

    parser.add_argument("--depth", type=int, default=1, help="Number of transformer blocks")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--mlp_dim", type=int, default=512, help="MLP hidden layer dimension")
    parser.add_argument("--dim_head", type=int, default=128, help="Dimension per attention head")
    parser.add_argument("--channels", type=int, default=33, help="Number of input channels")
    parser.add_argument("--patch_size", nargs=3, type=int, default=(3, 3, 3), help="Patch size (x y z)")
    parser.add_argument("--grid_size", nargs=3, type=int, default=(21, 21, 21), help="Grid size (x y z)")

    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for optimizer")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    training_data = GridDataset(args.train_data, args.grid_dir,
                                True)
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)


    CASF_data = GridDataset(args.casf_data, args.grid_dir)
    CASF_dataloader = DataLoader(CASF_data, batch_size=args.batch_size, shuffle=False)

    pdbbind_2013_core = GridDataset(args.core2013_data, args.grid_dir)
    pdbbind_2013_core_dataloader = DataLoader(pdbbind_2013_core, batch_size=args.batch_size, shuffle=False)


    validation_data = GridDataset(args.valid_data, args.grid_dir)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=False)
    


    model =ViT(args.grid_size, args.patch_size, depth=args.depth, heads=args.heads, 
               dim=args.dim, mlp_dim=args.mlp_dim, channels=args.channels)

    n_epoch = args.epochs
    batch_size = args.batch_size
    n_batches = len(training_data) // batch_size

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()

    loss_criterion = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
    best_val_loss = float('inf')
    losses = []
    for epoch in range(n_epoch):
        train_loss = 0.0
        valid_loss = 0.0
        valid_r2 = 0
        c = 0

        model.train()
        for train_data, train_labels in train_dataloader:
            c+=1

            optimizer.zero_grad()

            loss = loss_criterion(predictions.flatten(), train_labels.to(device).flatten())
            train_loss += loss.item()

            loss.backward()

            optimizer.step()

        model.eval()
        val_predictions = []
        val_true_labels = []

        CASF_pred = []
        CASF_true = []

        pdbbind_2013_core_pred = []
        pdbbind_2013_core_true = []

        with torch.no_grad():
            for eval_data, eval_labels  in validation_dataloader:
                target = model(eval_data.to(device)).flatten().detach().cpu().numpy()
                val_predictions.append(target)
                val_true_labels.append(eval_labels.flatten().detach().numpy())
                val_loss = loss_criterion(model(eval_data.to(device)).flatten(), eval_labels.to(device).flatten())
                valid_loss += val_loss.item()

            for eval_data, eval_labels in CASF_dataloader:
                target = model(eval_data.to(device)).flatten().detach().cpu().numpy()
                CASF_pred.append(target)
                CASF_true.append(eval_labels.flatten().detach().numpy())


            for eval_data, eval_labels in pdbbind_2013_core_dataloader:
                target = model(eval_data.to(device)).flatten().detach().cpu().numpy()
                pdbbind_2013_core_pred.append(target)
                pdbbind_2013_core_true.append(eval_labels.flatten().detach().numpy())

        val_predictions = np.concatenate(val_predictions)
        val_true_labels = np.concatenate(val_true_labels)
        val_mae = round(mean_absolute_error(val_true_labels, val_predictions), 3)
        val_pcc = round(pearsonr(val_true_labels, val_predictions)[0], 3)


        predictions = np.concatenate(CASF_pred)
        true_labels = np.concatenate(CASF_true)

        mae_CASF_2016 = round(mean_absolute_error(true_labels, predictions), 3)
        pcc_CASF_2016 = round(pearsonr(true_labels, predictions)[0], 3)
        rmse_CASF_2016 = round(root_mean_squared_error(true_labels, predictions), 3)

        predictions = np.concatenate(pdbbind_2013_core_pred)
        true_labels = np.concatenate(pdbbind_2013_core_true)

        mae_pdbbibd_2013 = round(mean_absolute_error(true_labels, predictions), 3)
        pcc_pdbbibd_2013 = round(pearsonr(true_labels, predictions)[0], 3)
        rmse_pdbbind_2013 = round(root_mean_squared_error(true_labels, predictions), 3)


        print(f"val: mae {val_mae} pcc {val_pcc}")
        print(f"CASF: mae {mae_CASF_2016} pcc {pcc_CASF_2016} rmse {rmse_CASF_2016}")
        print(f"pdbbind 2013: mae {mae_pdbbibd_2013} pcc {pcc_pdbbibd_2013} rmse {rmse_pdbbind_2013}")

        train_loss = train_loss / len(train_dataloader)
        valid_loss = valid_loss / len(validation_dataloader)

        print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'
              .format((epoch + 1), train_loss, valid_loss))

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), 'vit_model.pth')
            print(f'Best model saved at epoch {epoch + 1}')
