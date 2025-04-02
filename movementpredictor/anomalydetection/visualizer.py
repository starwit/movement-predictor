import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_input_target_output(x, y, mu, sigma, skew=None):
    frame_np = x[0].cpu().numpy()
    target = y.cpu().numpy()

    mask_others_np_sin = x[1].cpu().numpy()
    mask_others_np_cos = x[2].cpu().numpy()
    mask_others_np = np.zeros(frame_np.shape)
    mask_others_np[(mask_others_np_sin != 0) | (mask_others_np_cos != 0)] = 1

    mask_interest_np_sin = x[3].cpu().numpy()
    mask_interest_np_cos = x[4].cpu().numpy()
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
    cv2.circle(frame_rgb, [round(target[0]*frame_np.shape[-1]), round(target[1]*frame_np.shape[-2])], radius=2, color=(255, 0, 0), thickness=-1)
    plt.imshow(frame_rgb)
    if mask_others_np is not None:
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
    plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("prediction") if dist is None else plt.title("prediction, distance=" + str(dist))
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)

    if skew_lambda is None:
        plot_gaussian_variance(plt.gca(), frame_rgb, mu, sigma)
        '''
        eigenvalues, eigenvectors = np.linalg.eigh(sigma*frame_np.shape[-1]*0.1)  # add factor for better visualization
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        major_axis_length = 2 * np.sqrt(eigenvalues[0])  
        minor_axis_length = 2 * np.sqrt(eigenvalues[1]) 
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])) 

        circle = [round(mu[0]*frame_np.shape[-1]), round(mu[1]*frame_np.shape[-2])]
        cv2.circle(frame_rgb, circle, radius=2, color=(255, 0, 0), thickness=-1)
        
        cv2.ellipse(
            frame_rgb,
            center=circle,
            axes=(int(major_axis_length), int(minor_axis_length)),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=(255, 0, 0),
            thickness=1
        )
        '''

    else:
        plot_skewed_mahalanobis_points(plt.gca(), frame_rgb, mu, sigma, skew_lambda)

    plt.imshow(frame_rgb)
    if mask_others_np is not None:
        plt.imshow(mask_others_np, cmap='Reds', alpha=0.4, interpolation='nearest')
    plt.imshow(mask_interest_np, cmap='Blues', alpha=0.3, interpolation='nearest')
    plt.axis('off')


def plot_gaussian_variance(ax, frame_rgb, mu, sigma, scale_factor=0.1, num_points=100):
    """Plottet die symmetrische Gaussian-Varianz als Punktwolke."""
    
    height, width, _ = frame_rgb.shape
    angles = np.linspace(0, 2 * np.pi, num_points)
    
    # Einheitskreis-Punkte generieren
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)
    circle = np.stack([circle_x, circle_y], axis=1)
    
    # Kovarianz-Eigenwerte/-vektoren bestimmen
    eigenvalues, eigenvectors = np.linalg.eigh(sigma * width * scale_factor)
    
    # Skalierung durch Eigenwerte
    transformed_points = circle @ np.sqrt(np.diag(eigenvalues)) @ eigenvectors.T  
    points = mu + transformed_points * 0.5   

    # In Bild-Koordinaten umrechnen
    points[:, 0] *= width
    points[:, 1] *= height
    mu_scaled = (mu[0] * width, mu[1] * height)
    
    ax.imshow(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
    ax.scatter(points[:, 0], points[:, 1], c='cyan', s=5, label="Gaussian Variance Points")
    ax.scatter(mu_scaled[0], mu_scaled[1], color='red', marker='o', s=100, label="Mean")
    
    ax.legend()
    ax.set_title("Gaussian Variance")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')


def plot_skewed_mahalanobis_points(ax, frame_rgb, mu, sigma, skew_lambda, scale_factor=0.2, num_points=100):
    """Plottet Punkte mit gleicher skewed Mahalanobis-Distanz um den Mean."""
    
    height, width, _ = frame_rgb.shape
    angles = np.linspace(0, 2 * np.pi, num_points)
    
    # Einheitskreis-Punkte generieren
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)
    skew_factor_x = np.exp(-np.sign(circle_x) * skew_lambda[0])
    skew_factor_y = np.exp(-np.sign(circle_y) * skew_lambda[1])
    circle_x *= skew_factor_x
    circle_y *= skew_factor_y

    circle = np.stack([circle_x, circle_y], axis=1)

    # Kovarianz-Eigenwerte/-vektoren bestimmen
    eigvals, eigvecs = np.linalg.eigh(sigma * scale_factor)
    
    # Skalierung durch Eigenwerte
    transformed_points = circle @ np.sqrt(np.diag(eigvals)) @ eigvecs.T  
    
    # calculate skew factor: negative input because the circle points here are not the error but rather the opposite (allowed errors): vairance is larger in places of larger allowed error
    #skew_factors = np.exp(-np.sign(transformed_points) * skew_lambda)
    
    # Punkte in den Mahalanobis-Skalenraum transformieren
    #scaled_points = transformed_points * skew_factors 
    #points = mu + scaled_points   
    points = mu + transformed_points

    points[:, 0] *= width
    points[:, 1] *= height
    mu_scaled = (mu[0] * width, mu[1] * height)

    ax.imshow(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
    ax.scatter(points[:, 0], points[:, 1], c='cyan', s=5, label="Skewed Variance Points")
    ax.scatter(mu_scaled[0], mu_scaled[1], color='red', marker='o', s=100, label="Mean")

    ax.legend()
    ax.set_title("Skewed Mahalanobis Variance")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
