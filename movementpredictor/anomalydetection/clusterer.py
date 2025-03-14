import matplotlib.pyplot as plt
import numpy as np
import scipy
import hdbscan
from sklearn.decomposition import PCA
import logging
import os 
from tqdm import tqdm
import pybase64

from movementpredictor.data.datamanagement import get_downsampled_tensor_img
from movementpredictor.data.dataset import create_mask_tensor
from movementpredictor.anomalydetection.anomaly_detector import make_plot

from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump

log = logging.getLogger(__name__)


def get_clustering_vectors(a_inputs, a_targets, a_mus, a_probs):
    # 2 - center bbox input; 2 - target position; 2 - output mean; 1 - target prob 
    clustering_data = [[], [], [], [], [], [], []]

    for mu, p, inp, pos in zip(a_mus, a_probs, a_inputs, a_targets):
        center_x = (inp[0][0]+inp[1][0])/2
        center_y = (inp[0][1]+inp[1][1])/2
        clustering_data[0].append(center_x)
        clustering_data[1].append(center_y)
        #clustering_data[2].append(inp[1][0])
        #clustering_data[3].append(inp[1][1])

        clustering_data[2].append(pos[0])
        clustering_data[3].append(pos[1])
        
        clustering_data[4].append(mu[0])
        clustering_data[5].append(mu[1])

        clustering_data[6].append(p)
        #L = scipy.linalg.cholesky(cov, lower=True) 
        #clustering_data[8].append(L[0][0])
        #clustering_data[9].append(L[1][0])
        #clustering_data[10].append(L[1][1])

    # normalization parameter
    means = []
    stds = []

    normalized_data = []
    for row in clustering_data:
        mean = np.mean(row)
        std = np.std(row, ddof=0)  
        
        means.append(mean)
        stds.append(std)
        
        try:
            normalized_row = (row - mean) / std
        except:
            log.error("no std in :" + str(row[:10]))
        
        normalized_data.append(normalized_row)
    
    clustering_vectors = np.array(normalized_data).T

    return clustering_vectors, [means, stds]


def apply_clustering(cluster_data):
    hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=4, cluster_selection_epsilon=0.2, cluster_selection_method="eom", min_samples=4)
    clusters = hdbscan_cluster.fit_predict(cluster_data)
    print(clusters)

    # visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(cluster_data)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap="viridis")
    plt.colorbar(label='Cluster')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Clusters in 2dim space')
    plt.savefig("plots/Clusters_PCA.png")
    plt.clf()

    return clusters


def plot_anomalies_per_cluster(clusters, anomaly_inputs, anomaly_targets, anomaly_mus, anomaly_covs, anomaly_ts, path_sae_dump, dim_x, dim_y):
    max_cluster = max(clusters)
    outlier_counter = 1
    img_counter = np.zeros((max_cluster+1))
    
    with open(path_sae_dump, 'r') as input_file:
        messages = saedump.message_splitter(input_file)

        start_message = next(messages)
        saedump.DumpMeta.model_validate_json(start_message)

        anomaly_ts_int = [int(ts) for ts in anomaly_ts]

        for message in tqdm(messages, desc="plot cluster members"):
            event = saedump.Event.model_validate_json(message)
            proto_bytes = pybase64.standard_b64decode(event.data_b64)

            proto = SaeMessage()
            proto.ParseFromString(proto_bytes)
            frame_ts = proto.frame.timestamp_utc_ms

            if frame_ts in anomaly_ts_int:
                indices = [i for i, x in enumerate(anomaly_ts_int) if x == frame_ts]

                for index in indices:
                    frame_tensor = get_downsampled_tensor_img(proto.frame, dim_x, dim_y)
                    
                    c, inp, tar = clusters[index], anomaly_inputs[index], anomaly_targets[index]
                    mu, cov = anomaly_mus[index], anomaly_covs[index]

                    if c != -1:
                        path_cluster = "plots/clusters/nr" + str(c) + "/" 
                    else:
                        path_cluster = "plots/clusters/nr" + str(max_cluster+outlier_counter) + "/"
                        outlier_counter += 1

                    os.makedirs(path_cluster, exist_ok=True)

                    mask_interest_np = create_mask_tensor(dim_x, dim_y, [inp], scale=False).numpy()
                    make_plot(frame_tensor.numpy(), mask_interest_np, tar.cpu().numpy(), mu, cov)
                    
                    if c != -1:
                        plt.savefig(path_cluster + "point" + str(int(img_counter[c])) + ".png")
                        img_counter[c] += 1
                    else:
                        plt.savefig(path_cluster + "point.png")
                    plt.close()

