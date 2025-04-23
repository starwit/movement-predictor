import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

log = logging.getLogger(__name__)


def get_model(architecture, output_prob, path_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_prob == "symmetric":
        model = SymmetricProb(architecture=architecture)
    elif output_prob == "asymmetric":
        model = AsymmetricProb(architecture=architecture)
    else:
        log.error(f"{output_prob} is not a known ouput distribution setting. Is has to be 'symmetric' or 'asymmetric'")
        exit(1)

    if path_model is not None:
        weights = torch.load(path_model + "/model_weights.pth", map_location=device) 
        model.load_state_dict(weights, strict=True)

    model.to(device)
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
    backbone = models.mobilenet_v3_small(pretrained=True)

    old_conv = backbone.features[0][0]
    new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=old_conv.bias)
    
    backbone.features[0][0] = new_conv
    backbone_output_dim = backbone.classifier[3].in_features
    backbone.classifier = nn.Identity()

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

    def __init__(self, input_channels=5, architecture="SimpleCNN"):
        super(BaseProbabilistic, self).__init__()

        input_channels += 2     # coordinates channels

        if architecture=="ResNet18":
            self.backbone, backbone_output_dim = adapt_resnet18(input_channels)

        elif architecture=="MobileNet_v3":
            self.backbone, backbone_output_dim = adapt_mobilenet_v3(input_channels)
        
        elif architecture=="SimpleCNN":
            self.backbone = SimpleCNNBackbone(input_channels)
            backbone_output_dim = 512 * 4 * 4
        
        else: 
            log.error("architecture " + architecture + " does not exist!")

        self.fc = nn.Linear(backbone_output_dim, 64) 
        #self.fc2 = nn.Linear(256, 64)

        # output layer (distribution parameter)
        self.mean_layer = nn.Linear(64, 2)  # µ_x, µ_y
        self.log_var_layer = nn.Linear(64, 2)  # log(σ_x²), log(σ_y²)
        self.corr_layer = nn.Linear(64, 1)  # tanh(ρ)

    def forward_features(self, x):
        """feature extraction base"""
        x = BaseProbabilistic.add_coord_channels(x)

        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        #x = F.relu(self.fc2(x))
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
    
    @staticmethod
    def add_coord_channels(img_tensor):  # img_tensor: [B, C, H, W]
        B, C, H, W = img_tensor.shape
        y_coords = torch.linspace(-1, 1, steps=H, device=img_tensor.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, steps=W, device=img_tensor.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([img_tensor, x_coords, y_coords], dim=1)  # [B, C+2, H, W]


class SymmetricProb(BaseProbabilistic):
    def __init__(self, input_channels=5, architecture="SimpleCNN"):
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
    def __init__(self, input_channels=5, architecture="SimpleCNN"):
        super().__init__(input_channels, architecture)
        # additional layer for asymmetry
        self.skew_layer = nn.Linear(64, 2)  # λ_x, λ_y

    def forward(self, x):
        x = self.forward_features(x)
        mean, sigma = self.forward_common_outputs(x)
        skew_lambda = torch.tanh(self.skew_layer(x)) * 0.5  # Skewing ∈ (-0.5,0.5)
        return mean, sigma, skew_lambda
    
    @staticmethod
    def mahalanobis_distance(y_true, prediction, epsilon=1e-6):
        mu, sigma, skew_lambda = prediction
        error = (y_true - mu).unsqueeze(2)  # Shape (batch_size, 2, 1)
        
        sigma_stable = sigma + epsilon * torch.eye(sigma.size(-1)).to(sigma.device) 
        sigma_inv = torch.inverse(sigma_stable)
        mahalanobis = torch.bmm(torch.bmm(error.transpose(1, 2), sigma_inv), error)

        error = error.squeeze(2)

        s_x = torch.exp(torch.sign(error[:, 0]) * skew_lambda[:, 0])
        s_y = torch.exp(torch.sign(error[:, 1]) * skew_lambda[:, 1])

        skew_factor = torch.stack([s_x, s_y], dim=1)
        skewed_mahalanobis = mahalanobis * skew_factor.sum(dim=1, keepdim=True).unsqueeze(-1)

        return skewed_mahalanobis, sigma_stable

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
        loss = loss + + regularization_term(sigma_stable, y_true, slope, intercept)

        return loss.mean()
    
