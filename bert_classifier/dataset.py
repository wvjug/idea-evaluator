from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd 
import torch

class HackathonDataset(Dataset):
    """custom dataset class that reads in {problem, solution} pairs"""
    def __init__(self, data_path: Path):
        self.data = pd.read_csv(data_path, encoding="latin1")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        problem = self.data.loc[idx, "Problem"]
        solution = self.data.loc[idx, "Solution"]

        # concatenate the scores from three criteria 
        # label = self.data.loc[idx, ["Circular Economy", "Market Potentials", "Feasibility"]]
        label = self.data.loc[idx, "Feasibility"]

        # return: (str) {problem, solution} pair, (Tensor[float]) ordering score from 1-5 for three criteria
        return "Problem: " + str(problem) + "\nSolution: " + str(solution), torch.tensor(label).to(torch.long)*2-2
