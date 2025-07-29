import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from movementpredictor.cnn.model_architectures import AsymmetricProb
from matplotlib.lines import Line2D


def plot_input_target_output(frame, x, y, mu, sigma, skew=None):
    frame_np = frame.numpy()
    target = y.cpu().numpy()

    mask_others_np_sin = x[0].cpu().numpy()
    mask_others_np_cos = x[1].cpu().numpy()
    mask_others_np = np.zeros(frame_np.shape)
    mask_others_np[(mask_others_np_sin != 0) | (mask_others_np_cos != 0)] = 1

    mask_interest_np_sin = x[-2].cpu().numpy()
    mask_interest_np_cos = x[-1].cpu().numpy()
    mask_interest_np = np.zeros(frame_np.shape)
    mask_interest_np[(mask_interest_np_sin != 0) | (mask_interest_np_cos != 0)] = 1
        
    # calculate angle
    sin = np.max(mask_interest_np_sin) if np.max(mask_interest_np_sin) > 0 else np.min(mask_interest_np_sin)
    cos = np.max(mask_interest_np_cos) if np.max(mask_interest_np_cos) > 0 else np.min(mask_interest_np_cos)
    angle_rad = math.atan2(sin, cos)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    angle_deg = round(angle_deg/2)

    make_plot(frame_np, mask_interest_np, target, mu, sigma, mask_others_np, angle=angle_deg, skew_lambda=skew)


def make_plot(frame_np, mask_interest_np, target, mu, sigma, mask_others_np=None, angle=None, dist=None, skew_lambda=None):
    plt.figure(figsize=(22, 7))

    plt.subplot(1, 3, 1)
    plt.title("input") if angle is None else plt.title("input, orientation angle: " + str(angle))
    plt.imshow(frame_np, cmap='gray', interpolation='nearest')
    if mask_others_np is not None:
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
    plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("target")
    frame_np = (frame_np * 255).astype(np.uint8)
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
    plt.imshow(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
    plt.scatter(round(target[0]*frame_np.shape[-1]), round(target[1]*frame_np.shape[-2]), color='red', marker='x', s=100, label="Actual position " + r'$C_{t+\delta,j}$')

    if mask_others_np is not None:
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
    plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
    plt.axis('off')

    plt.legend(fontsize=16, loc="lower right")

    plt.subplot(1, 3, 3)
    plt.title("prediction") if dist is None else plt.title("prediction, distance=" + str(dist))
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)

    if skew_lambda is None:
        plot_gaussian_variance(plt.gca(), frame_rgb, mu, sigma)
    else:
        plot_skewed_mahalanobis_points(plt.gca(), frame_rgb, mu, sigma, skew_lambda)

    plt.imshow(frame_rgb)
    if mask_others_np is not None:
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
    plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
    plt.axis('off')


def plot_gaussian_variance(ax, frame_rgb, mu, sigma, scale_factor=1, num_points=100):
    """Plottet die symmetrische Gaussian-Varianz als Punktwolke."""
    
    height, width, _ = frame_rgb.shape
    angles = np.linspace(0, 2 * np.pi, num_points)
    
    # generate points on unit circle
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)
    circle = np.stack([circle_x, circle_y], axis=1)
    
    eigenvalues, eigenvectors = np.linalg.eigh(sigma * scale_factor)
    
    # skale points with eigenvalues
    transformed_points = circle @ np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T  
    points = mu + transformed_points * 0.5   

    # get the frame coordiantes
    points[:, 0] *= width
    points[:, 1] *= height
    mu_scaled = (mu[0] * width, mu[1] * height)
    
    ax.imshow(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
    ax.scatter(points[:, 0], points[:, 1], c='cyan', s=5, label="Gaussian variance " + r'$\Sigma_{t+\delta,j}$')
    ax.scatter(mu_scaled[0], mu_scaled[1], color='red', marker='o', s=100, label="Expected position " + r'$\hat{C}_{t+\delta,j}$')

    proxy_line = Line2D([0], [0],
                    color='cyan',
                    linestyle='-',
                    linewidth=2,
                    label=r'Gaussian variance $\Sigma_{t+\delta,j}$')
    handles, labels = ax.get_legend_handles_labels()
    for i, lab in enumerate(labels):
        if 'Gaussian variance' in lab:
            handles[i] = proxy_line
    
    ax.legend(handles=handles, fontsize=16, loc="lower right")
    ax.set_title("Gaussian Variance")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')


def find_point_on_loss_contour(mu_np, sigma_np, skew_lambda_np, direction, loss_target=0.2, max_iter=30):
    mu = torch.tensor(mu_np, dtype=torch.float32)
    sigma = torch.tensor(sigma_np, dtype=torch.float32)
    skew_lambda = torch.tensor(skew_lambda_np, dtype=torch.float32)

    s_min, s_max = 0.0, 1.0
    direction = direction / np.linalg.norm(direction)

    for _ in range(max_iter):
        s_mid = (s_min + s_max) / 2
        y = mu + s_mid * torch.tensor(direction, dtype=torch.float32)
        loss_val, _ = AsymmetricProb.mahalanobis_distance(
            y.unsqueeze(0),
            (mu.unsqueeze(0), sigma.unsqueeze(0), skew_lambda.unsqueeze(0))
        )
        if abs(loss_val.item() - loss_target) < 1e-4:
            break
        if loss_val.item() > loss_target:
            s_max = s_mid
        else:
            s_min = s_mid

    return (mu + s_mid * torch.tensor(direction, dtype=torch.float32)).numpy()


def plot_skewed_mahalanobis_points(ax, frame_rgb, mu, sigma, skew_lambda, scale_factor=0.3, num_points=100):
    """Plottet Punkte mit gleicher skewed Mahalanobis-Distanz um den Mean."""
    
    height, width, _ = frame_rgb.shape
    angles = np.linspace(0, 2 * np.pi, num_points)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    points = []

    for dir_vec in directions:
        pt = find_point_on_loss_contour(mu, sigma, skew_lambda, dir_vec, loss_target=scale_factor)
        pt_scaled = [pt[0] * width, pt[1] * height]
        points.append(pt_scaled)
    
    points = np.array(points)
    mu_scaled = (mu[0] * width, mu[1] * height)

    ax.imshow(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
    ax.plot(points[:, 0], points[:, 1], color="cyan", linewidth=3, label=r'skewed $D_M$ boundary')
    ax.scatter(mu_scaled[0], mu_scaled[1], color='red', marker='o', s=100, label="Expected position " + r'$\hat{C}_{t+\delta,j}$')

    ax.legend(fontsize=16, loc="lower right")
    ax.set_title("Skewed Mahalanobis Variance")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')