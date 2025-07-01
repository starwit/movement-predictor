import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
import os

log = logging.getLogger(__name__)


def get_model(architecture, output_prob, path_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_prob == "symmetric":
        model = SymmetricProb(architecture=architecture)
    elif output_prob == "asymmetric":
        model = AsymmetricProb(architecture=architecture)
    else:
        log.error(f"{output_prob} is not a known ouput distribution setting. Is has to be 'symmetric' or 'asymmetric'")
        return None
    
    if architecture not in ["SimpleCNN", "MobileNet_v3", "SwinTransformer", "ResNet18"]:
        log.error(f"{architecture} is not a known model architecture. Is has to be 'SimpleCNN', 'MobileNet_v3', 'SwinTransformer' or 'ResNet18'")
        return None

    if path_model is not None:
        if path_model.endswith(".pth") and os.path.isfile(path_model):
            weight_path = path_model
        else:
            weight_path = os.path.join(path_model, "model_weights.pth")

        try:
            weights = torch.load(weight_path, map_location=device)
        except:
            log.error(f"Model weights not found at {weight_path}")
            return None
        try:    
            model.load_state_dict(weights, strict=True)
        except:
            log.error(f"Model weights do not fit model architecture")
            return None

    model.to(device)
    model.eval()
    return model


def regularization_term(sigma, y_true, slope, intercept):
    trace_sigma = torch.einsum("bii->b", sigma) 
    y = y_true[:, 1]
    scaling_factor = slope*y + intercept     # slope*y + intercept = width + height bounding box  ->  higher scaling factor for smaller bbox (cars further away)
    max_val = torch.max(scaling_factor).detach()
    scaling_factor = (max_val - scaling_factor)/max_val + 0.1
    return scaling_factor*torch.log1p(trace_sigma)


def adapt_resnet18(input_channels):
    backbone = models.resnet18() 

    old_conv = backbone.conv1
    new_conv = nn.Conv2d(input_channels, old_conv.out_channels, 
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, 
                         padding=old_conv.padding, 
                         bias=old_conv.bias)
    
    backbone.conv1 = new_conv
    backbone_output_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    return backbone, backbone_output_dim


def adapt_mobilenet_v3(input_channels):
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    old_conv = backbone.features[0][0]
    new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=old_conv.bias)
    
    backbone.features[0][0] = new_conv
    backbone_output_dim = backbone.classifier[0].in_features
    backbone.classifier = nn.Identity()

    return backbone, backbone_output_dim


def adapt_swin_transformer(input_channels):
    backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

    old_conv = backbone.features[0][0]  
    new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding)

    backbone.features[0][0] = new_conv
    backbone_output_dim = backbone.head.in_features
    backbone.head = nn.Identity()

    return backbone, backbone_output_dim


class SimpleCNNBackbone(nn.Module):
    def __init__(self, input_channels):
        super(SimpleCNNBackbone, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)  # -> 60x60
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> 30x30
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> 15x15
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # -> 8x8
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # -> 4x4

    def forward(self, x):
        """feature extraction base"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


class BaseProbabilistic(nn.Module):

    def __init__(self, input_channels=4, architecture="SimpleCNN"):
        super(BaseProbabilistic, self).__init__()

        self.architecture = architecture

        if architecture=="ResNet18":
            self.backbone, self.backbone_output_dim = adapt_resnet18(input_channels)

        elif architecture=="MobileNet_v3":
            self.backbone, self.backbone_output_dim = adapt_mobilenet_v3(input_channels)
        
        elif architecture =="SwinTransformer":
            self.backbone, self.backbone_output_dim = adapt_swin_transformer(input_channels)
        
        elif architecture=="SimpleCNN":
            self.backbone = SimpleCNNBackbone(input_channels)
            self.backbone_output_dim = 512 * 4 * 4
        
        else: 
            log.error("Model architecture " + architecture + " does not exist!")
            exit(1)

        self.mean_layer = nn.Linear(self.backbone_output_dim, 2)  # µ_x, µ_y
        self.log_var_layer = nn.Linear(self.backbone_output_dim, 2)  # log(σ_x²), log(σ_y²)
        self.corr_layer = nn.Linear(self.backbone_output_dim, 1)  # tanh(ρ)

    def forward_features(self, x):
        """feature extraction base"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

    def forward_common_outputs(self, x):
        """calculate parameter of bivariate gaussian distribution"""
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        corr = torch.tanh(self.corr_layer(x))

        var_x = torch.exp(log_var[:, 0])  
        var_y = torch.exp(log_var[:, 1])  
        cov_xy = corr[:, 0] * torch.sqrt(var_x * var_y)  

        sigma = torch.stack([var_x, cov_xy, cov_xy, var_y], dim=1).view(-1, 2, 2)
        return mean, sigma


class SymmetricProb(BaseProbabilistic):
    def __init__(self, input_channels=4, architecture="SimpleCNN"):
        super().__init__(input_channels, architecture)

    def forward(self, x):
        x = self.forward_features(x)
        mean, sigma = self.forward_common_outputs(x)
        return mean, sigma
    
    @staticmethod
    def mahalanobis_distance(y_true, prediction, epsilon=1e-6):
        mu, sigma = prediction
        error = (y_true - mu).unsqueeze(2)  # Shape (batch_size, 2, 1)
        
        sigma_stable = sigma + epsilon * torch.eye(sigma.size(-1)).to(sigma.device) 
        sigma_inv = torch.inverse(sigma_stable)

        mahalanobis = torch.bmm(torch.bmm(error.transpose(1, 2), sigma_inv), error)
        return mahalanobis, sigma_stable
    
    @staticmethod
    def loss(y_true, prediction, slope, intercept):
        """
        Negative Log-Likelihood (NLL) Loss with regularization
        
        Args:
            y_true (torch.Tensor): true future position (batch_size, 2)
            prediction: mean (torch.Tensor, (batch_size, 2)), covariance matrix (torch.Tensor, (batch_size, 2, 2))
            
        Returns:
            torch.Tensor: mean nll-loss of whole batch.
        """
        epsilon = 1e-6  
        mahalanobis, sigma_stable = SymmetricProb.mahalanobis_distance(y_true, prediction, epsilon) 
        loss = mahalanobis.squeeze() + regularization_term(sigma_stable, y_true, slope, intercept)

        return loss.mean()



class AsymmetricProb(BaseProbabilistic):
    def __init__(self, input_channels=4, architecture="SimpleCNN"):
        super().__init__(input_channels, architecture)
        # additional layer for asymmetry
        self.skew_layer = nn.Linear(self.backbone_output_dim, 2)  # λ_x, λ_y

    def softsign(x):
        return x / (1 + x.abs())

    def forward(self, x):
        x = self.forward_features(x)
        mean, sigma = self.forward_common_outputs(x)
        skew_lambda = torch.tanh(self.skew_layer(x))  * 0.3                           #torch.tanh(self.skew_layer(x)) * 0.3 # Skewing ∈ (-1, 1)
        return mean, sigma, skew_lambda
    
    def mahalanobis_distance(y_true, prediction, epsilon=1e-6):
        mu, sigma, skew_lambda = prediction
        error = (y_true - mu)  

        sigma_stable = sigma + epsilon * torch.eye(2, device=sigma.device)
        sigma_inv = torch.inverse(sigma_stable)

        skew_factor = torch.exp(torch.sign(error) * skew_lambda)  # (B, 2)
        error_skewed = error * skew_factor  # (B, 2)

        err = error_skewed.unsqueeze(1)  # (B,1,2)
        maha = torch.bmm(torch.bmm(err, sigma_inv), err.transpose(1, 2))

        return maha, sigma_stable

    @staticmethod
    #https://pmc.ncbi.nlm.nih.gov/articles/PMC7615262/pdf/tmi-li-3231730.pdf
    def loss(y_true, prediction, slope, intercept):
        """
        Skewed Negative Log-Likelihood (NLL) Loss with regularization
        
        Args:
            y_true (torch.Tensor): true future position (batch_size, 2)
            prediction: mean (torch.Tensor, (batch_size, 2)), covariance matrix (torch.Tensor, (batch_size, 2, 2)), skew parameters lambda (torch.Tensor, (batch_size, 2))
            
        Returns:
            torch.Tensor: mean nll-loss of whole batch.
        """
        epsilon = 1e-6  
        skewed_mahalanobis, sigma_stable = AsymmetricProb.mahalanobis_distance(y_true, prediction, epsilon)
        loss = skewed_mahalanobis.squeeze() 
        skew_penalty = (prediction[-1] ** 2).mean()
        loss = loss + regularization_term(sigma_stable, y_true, slope, intercept) + 0.01 * skew_penalty

        return loss.mean()