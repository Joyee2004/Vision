from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
from vocab import Vocabulary
from PIL import Image
import torch



class FlickrDataset(Dataset):
    
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        
        valid_indices = []
        for i in range(len(self.imgs)):
            img_name = self.imgs[i]
            img_location = os.path.join(self.root_dir, img_name)
            if os.path.exists(img_location):
                valid_indices.append(i)

        self.imgs = self.imgs.iloc[valid_indices].reset_index(drop=True)
        self.captions = self.captions.iloc[valid_indices].reset_index(drop=True)

        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
        
        self.image_sizes = self._find_image_sizes()

    def _find_image_sizes(self):
        sizes = set()
        for img_name in self.imgs:
            img_location = os.path.join(self.root_dir, img_name)
            with Image.open(img_location) as img:
                sizes.add(img.size)
        return sizes

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)

       
        print(f"Loading image from: {img_location}")

        img = Image.open(img_location).convert("RGB")

        
        if self.transform is not None:
            img = self.transform(img)

    
        caption_vec = []
        caption_vec += [self.vocab.stoi["<start>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<end>"]]

        return img, torch.tensor(caption_vec)
        
