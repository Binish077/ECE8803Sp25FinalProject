import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch

# checking for texture feature extraction libraries
try:
    from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

def extract_texture_features(image_arr):
    """
    function that extracts texture information (GLCM, LBP) from grayscale images using the skimage libraries.
    """
    feats = [] # storing texture information
    if not SKIMAGE_AVAILABLE:
        return np.zeros(12, dtype=np.float32)
    
    # GLCM features
    glcm = greycomatrix(image_arr, [1], [0], 256, symmetric=True, normed=True)
    feats.append(greycoprops(glcm, 'contrast')[0, 0])
    feats.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    feats.append(greycoprops(glcm, 'homogeneity')[0, 0])
    feats.append(greycoprops(glcm, 'ASM')[0, 0])
    feats.append(greycoprops(glcm, 'energy')[0, 0])
    feats.append(greycoprops(glcm, 'correlation')[0, 0])
    

    lbp = local_binary_pattern(image_arr, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # only observing the top 6 patterns
    feats.extend(hist[:6])
    return np.array(feats, dtype=np.float32)

class OLIVES(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms

        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx,0]

        # Image data
        image = Image.open(path).convert("L")
        image_arr = np.array(image)
        image_tensor = self.transforms(Image.fromarray(image_arr))

        # Clinical Data from Training_Unlabeled_Data.csv
        clinical_feats = np.zeros(4, dtype=np.float32)
        clinical_cols = ['BCVA', 'CST', 'Eye_ID', 'Patient_ID']
        if all(col in self.df.columns for col in clinical_cols):
            vals = self.df.loc[idx, clinical_cols].to_numpy(dtype=np.float32)
            clinical_feats[:len(vals)] = vals
        clinical_tensor = torch.tensor(clinical_feats, dtype=torch.float32)

        # Textural features
        texture_feats = extract_texture_features(image_arr)
        texture_tensor = torch.tensor(texture_feats, dtype=torch.float32)

        # Biomarker Labels
        b1 = self.df.iloc[idx,1]
        b2 = self.df.iloc[idx,2]
        b3 = self.df.iloc[idx,3]
        b4 = self.df.iloc[idx, 4]
        b5 = self.df.iloc[idx, 5]
        b6 = self.df.iloc[idx, 6]
        bio_tensor = torch.tensor([b1, b2, b3, b4, b5, b6], dtype=torch.float32)

        return image_tensor, clinical_tensor, texture_tensor, bio_tensor


class RECOVERY(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.df.iloc[idx, 0])

        # error handling
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path {path} does not exist")

        # image data
        image = Image.open(path).convert("L")
        image_arr = np.array(image)
        image_tensor = self.transforms(Image.fromarray(image_arr))

        # Gather clinical data for RECOVERY dataset
        clinical_feats = np.zeros(4, dtype=np.float32)
        clinical_cols = ['BCVA', 'CST', 'Eye_ID', 'Patient_ID']
        if all(col in self.df.columns for col in clinical_cols):
            vals = self.df.loc[idx, clinical_cols].to_numpy(dtype=np.float32)
            clinical_feats[:len(vals)] = vals
        clinical_tensor = torch.tensor(clinical_feats, dtype=torch.float32)

        # Gather texture data if possible
        if SKIMAGE_AVAILABLE:
            texture_feats = extract_texture_features(image_arr)
        else:
            texture_feats = np.zeros(12, dtype=np.float32)
        texture_tensor = torch.tensor(texture_feats, dtype=torch.float32)
        
        # empty array for biomarker labels
        bio_tensor = torch.zeros(6, dtype=torch.float32)

        return image_tensor, clinical_tensor, texture_tensor, bio_tensor