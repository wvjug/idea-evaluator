from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd 
import torch

class PSDataset(Dataset):
    def __init__(self, data_path: Path):
        self.data = pd.read_csv(data_path, encoding="latin1")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        problem = self.data.loc[idx, "Problem"]
        solution = self.data.loc[idx, "Solution"]
        label = self.data.loc[idx, ["Circular Economy", "Market Potentials", "Feasibility"]]
        return problem, solution, torch.tensor(label) 
    

if __name__ == "__main__":
    data_path = Path("/home/mingchiehliu/ai_earthhack/test.csv")
    d = PSDataset(data_path)
    for i in range(len(d)):
        problem, solution, label = d[i]
        # print(label)