from movementpredictor.data.datamanagement import TrackingDataManager
from movementpredictor.data.datafilterer import DataFilterer
from movementpredictor.config import ModelConfig
from movementpredictor.data import dataset

import os
import logging

log = logging.getLogger(__name__)
config = ModelConfig()


def main():
    trackManager = TrackingDataManager()
    trackedObjects = trackManager.getTrackedBaseData(config.path_sae_data, inferencing=True)
    trackedObjects = DataFilterer().apply_filtering(trackedObjects) 
    dataset.store_data(trackedObjects, config.path_store_data, trackManager.frame_rate, check_if_exist=True)


if __name__ == "__main__":
    main()