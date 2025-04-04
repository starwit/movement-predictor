import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseProbabilistic(nn.Module):
    def __init__(self, input_channels=5):
        super(CNNBaseProbabilistic, self).__init__()

        # CNN: Feature Extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)  # -> 60x60
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> 30x30
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> 15x15
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # -> 8x8
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # -> 4x4

        self.fc1 = nn.Linear(512 * 4 * 4, 256)  # Flattened Features
        self.fc2 = nn.Linear(256, 64)

        # output layer (distribution parameter)
        self.mean_layer = nn.Linear(64, 2)  # µ_x, µ_y
        self.log_var_layer = nn.Linear(64, 2)  # log(σ_x²), log(σ_y²)
        self.corr_layer = nn.Linear(64, 1)  # tanh(ρ)

    def forward_features(self, x):
        """feature extraction base"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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


class CNN_symmetric_prob(CNNBaseProbabilistic):
    def __init__(self, input_channels=5):
        super().__init__(input_channels)

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
    def loss(y_true, prediction):
        """
        Negative Log-Likelihood (NLL) Loss with regularization
        
        Args:
            y_true (torch.Tensor): true future position (batch_size, 2)
            prediction: mean (torch.Tensor, (batch_size, 2)), covariance matrix (torch.Tensor, (batch_size, 2, 2))
            
        Returns:
            torch.Tensor: mean nll-loss of whole batch.
        """
        epsilon = 1e-6  
        mahalanobis, sigma_stable = CNN_symmetric_prob.mahalanobis_distance(y_true, prediction, epsilon) 
        trace_sigma = torch.einsum("bii->b", sigma_stable) 
        mahalanobis_scaled = mahalanobis * (trace_sigma + epsilon)
        #TODO: + trace anstatt * trace ausprobieren!!!!!

        loss = mahalanobis_scaled.squeeze() 

        return loss.mean()


class CNN_asymmetric_prob(CNNBaseProbabilistic):
    def __init__(self, input_channels=5):
        super().__init__(input_channels)
        # additional layer for asymmetry
        self.skew_layer = nn.Linear(64, 2)  # λ_x, λ_y

    def forward(self, x):
        x = self.forward_features(x)
        mean, sigma = self.forward_common_outputs(x)
        skew_lambda = torch.tanh(self.skew_layer(x)) * 0.2  # Skewing ∈ (-0.2,0.2)
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
    def loss(y_true, prediction):
        """
        Skewed Negative Log-Likelihood (NLL) Loss with regularization
        
        Args:
            y_true (torch.Tensor): true future position (batch_size, 2)
            prediction: mean (torch.Tensor, (batch_size, 2)), covariance matrix (torch.Tensor, (batch_size, 2, 2)), skew parameters lambda (torch.Tensor, (batch_size, 2))
            
        Returns:
            torch.Tensor: mean nll-loss of whole batch.
        """
        epsilon = 1e-6  
        skewed_mahalanobis, sigma_stable = CNN_asymmetric_prob.mahalanobis_distance(y_true, prediction, epsilon)
        loss = skewed_mahalanobis.squeeze() 

        trace_sigma = torch.einsum("bii->b", sigma_stable) 
        loss = loss + torch.log1p(trace_sigma)

        return loss.mean()