import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from torchvision.models import efficientnet_b0

# Checking if we can use libraries to extract texture features. If not, we won't use texture in our models
try:
    from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class ResNet(nn.Module):
    """
    Baseline ResNet model provided in starting kit. There are 2 modes: resnet50 and resnet18
    Features:
        Resnet base
        encoder and decoder
    """
    def __init__(self, name='resnet50', num_classes=6):
        super(ResNet, self).__init__()
        if (name == 'resnet50'):
            self.encoder = torchvision.models.resnet50(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(2048, num_classes)
        else:
            self.encoder = torchvision.models.resnet18(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(512, num_classes)

class EnhancedResnet50(nn.Module):
    """
    Enhanched ResNet50 model with additional convolutional layers for feature extraction and dropout.
    Features:
      - ResNet50 backbone
      - Additional 2D convolutional layers for feature extraction
      - Dropout for regularization
    """
    def __init__(self, num_biomarkers=6, pretrained=True):
        super(PersonalizedBiomarkerPredictor, self).__init__()
        
        # base ResNet50 model
        self.base = models.resnet50(pretrained=pretrained)
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base.fc = nn.Identity()
        
        # feature extraction convolution layers
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        # final classification layer
        self.fc = nn.Linear(512, num_biomarkers)
        
        # dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base(x)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1) 

        x = self.dropout(x) 
        x = self.fc(x)
        return x

class MMT(nn.Module):
    """
    Multi-modal model that merges image data with textural information to predict biomarkers.
    Features:
      - EfficientNet for grayscale images
      - Optional MLP for texture
      - Fusion block to combine features
    """
    def __init__(self, num_biomarkers=6, texture_dim=12, use_texture=True):
        super().__init__()

        # EfficientNet50-b0 base modified for grayscale images
        self.base = efficientnet_b0(pretrained=True)
        self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.img_feat_dim = self.base.classifier[1].in_features

        # texture MLP block
        self.use_texture = use_texture and SKIMAGE_AVAILABLE
        if self.use_texture:
            self.texture_mlp = nn.Sequential(
                nn.Linear(texture_dim, 16),
                nn.ReLU()
            )
            self.texture_feat_dim = 16
        else:
            self.texture_feat_dim = 0

        # fusion block
        fusion_dim = self.img_feat_dim + self.texture_feat_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_biomarkers)
        )

    def forward(self, image, clinical, texture=None):
        
        # base
        img_feat = self.base.features(image)
        img_feat = self.base.avgpool(img_feat)
        img_feat = torch.flatten(img_feat, 1)

        # texture
        if self.use_texture and texture is not None:
            texture_feat = self.texture_mlp(texture)
        else:
            texture_feat = None

        # fusion
        feats = [img_feat, clinical_feat]
        if texture_feat is not None:
            feats.append(texture_feat)
        x = torch.cat(feats, dim=1)
        out = self.fusion_fc(x)
        return out

class MMCT(nn.Module):
    """
    Multi-modal model that merges both clinical data and textural information to predict biomarkers.
    Features:
      - EfficientNet for images
      - MLP for clinical data
      - Optional MLP for texture
      - Fusion block to combine features
    """
    def __init__(self, num_biomarkers=6, clinical_dim=4, texture_dim=12, use_texture=True):
        super().__init__()
        
        # EfficientNet backbone for grayscale images
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.img_feat_dim = self.backbone.classifier[1].in_features
        
        # clincal features block (BCVA, CST, Eye_ID, Patient_ID)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.clinical_feat_dim = 16
 
        # texture block
        self.use_texture = use_texture and SKIMAGE_AVAILABLE
        if self.use_texture:
            self.texture_mlp = nn.Sequential(
                nn.Linear(texture_dim, 16),
                nn.ReLU()
            )
            self.texture_feat_dim = 16
        else:
            self.texture_feat_dim = 0
 
        # fusion block
        fusion_dim = self.img_feat_dim + self.clinical_feat_dim + self.texture_feat_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_biomarkers)
        )
 
    def forward(self, image, clinical, texture=None):
        
        # image block
        img_feat = self.backbone.features(image)
        img_feat = self.backbone.avgpool(img_feat)
        img_feat = torch.flatten(img_feat, 1)

        # Clinical block
        clinical_feat = self.clinical_mlp(clinical)
        
        # texture block (optional)
        if self.use_texture and texture is not None:
            texture_feat = self.texture_mlp(texture)
        else:
            texture_feat = None
        
        # fusion
        feats = [img_feat, clinical_feat]
        if texture_feat is not None:
            feats.append(texture_feat)
        x = torch.cat(feats, dim=1)
        out = self.fusion_fc(x)
        return out

if __name__ == "__main__":
    
    print("Model loaded successfully.")

