"""
CelebA dataset reader that works with both the official layout
(list_attr_celeba.txt) and the Kaggle one (list_attr_celeba.csv, nested
img_align_celeba/img_align_celeba/ folder).

Returns (image, [male, smiling, young]) with attributes in {0, 1}.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.filenames = []
        self.all_attributes = []

        # Look for the CSV or TXT file
        csv_path = os.path.join(root_dir, "list_attr_celeba.csv")
        txt_path = os.path.join(root_dir, "list_attr_celeba.txt")

        if os.path.exists(csv_path):
            print(f"Loading attributes from CSV: {csv_path}")
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                # Kaggle CSV: Row 0 is the header. Data from row 1 onwards.
                # Example row: "000001.jpg,-1,1,..."
                data_lines = lines[1:]
                separator = ','
        
        elif os.path.exists(txt_path):
            print(f"Loading attributes from TXT: {txt_path}")
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                # CelebA TXT: Row 0 is the count, Row 1 is the header. Data from row 2.
                data_lines = lines[2:]
                separator = None # split() without arguments handles multiple spaces automatically
        
        else:
            raise FileNotFoundError(f"Cannot find list_attr_celeba (.csv or .txt) in {root_dir}")

        # Line parsing
        for line in data_lines:
            line = line.strip()
            if not line: continue
            
            if separator:
                parts = line.split(separator)
            else:
                parts = line.split() # Handles multiple spaces in txt
            
            # parts[0] is the filename
            self.filenames.append(parts[0])
            
            # parts[1:] are the attributes (converted to int)
            # Note: sometimes the csv has empty values at the end, we take the first 40 attrs to be safe
            attrs = [int(x) for x in parts[1:41]] 
            self.all_attributes.append(attrs)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 1. Load Image
        img_name = self.filenames[idx]
        
        # Handle nested or flat paths (Kaggle dataset structure vs standard)
        img_path_nested = os.path.join(self.root_dir, 'img_align_celeba', 'img_align_celeba', img_name)
        img_path_flat = os.path.join(self.root_dir, 'img_align_celeba', img_name)
        
        if os.path.exists(img_path_nested):
            img_path = img_path_nested
        else:
            img_path = img_path_flat

        image = Image.open(img_path).convert('RGB')

        # 2. Transformations
        if self.transform:
            image = self.transform(image)

        # 3. Extract Attributes
        row_attrs = self.all_attributes[idx]

        # Fixed CelebA indices: 20=Male, 31=Smiling, 39=Young
        # Conversion formula: (value + 1) // 2  ->  (-1 becomes 0, 1 becomes 1)
        
        male = (row_attrs[20] + 1) // 2
        smiling = (row_attrs[31] + 1) // 2
        young = (row_attrs[39] + 1) // 2

        target_attrs = torch.tensor([male, smiling, young], dtype=torch.float32)

        return image, target_attrs
