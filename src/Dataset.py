from pathlib import Path
from PIL import Image
import pandas as pd


# for training set, always returns labels (has_labels = True)
# for testing set, no labels (has_labels = False)
class Dataset:

    def __init__(self, labels, csv_path, img_dir, has_labels=False, transform=None):
        
        self.has_labels = has_labels
        self.img_dir = img_dir

        self.df = pd.read_csv(csv_path)

        self.label_dict = {}

        for i, label in enumerate(labels):
            self.label_dict[label] = i
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        filename = row['bcn_filename']

        full_path = Path(self.img_dir) / filename

        img = Image.open(full_path)
        transform_img = self.transform(img)

        if (self.has_labels):
            diagnosis = row['diagnosis']
        else:
            return (transform_img, -1)
        
        num_label = self.label_dict[diagnosis]
        
        return (transform_img, num_label)
        



        



