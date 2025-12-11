import torch
from torch.utils.data import Dataset

class TokenJourneyDataset(Dataset):
    def __init__(self, pt_file):
        """
        Args:
            pt_file (str): Path to processed .pt file (train_data.pt or test_data.pt)
        """
        print(f"Loading dataset from {pt_file}...")
        data = torch.load(pt_file)
        
        self.layers = data['layers']
        self.history = data['history']
        self.secondary = data['secondary']
        self.gating = data['gating']
        self.pos = data['pos']
        self.targets = data['targets']
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'layer': self.layers[idx],
            'history': self.history[idx],
            'secondary': self.secondary[idx],
            'gating': self.gating[idx],
            'pos': self.pos[idx],
            'target': self.targets[idx]
        }
