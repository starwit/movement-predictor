from movementpredictor.data.datamanagement import TrackingDataManager
from movementpredictor.data.datafilterer import DataFilterer
from movementpredictor.config import ModelConfig
from movementpredictor.data import dataset, datamanagement

import os
import logging

log = logging.getLogger(__name__)
config = ModelConfig()


def main():
    trackManager = TrackingDataManager()
    trackedObjects = trackManager.getTrackedBaseData(config.path_sae_data_test, inferencing=True)
    trackedObjects = DataFilterer().apply_filtering(trackedObjects)

    name_sae_dump =  os.path.basename(config.path_sae_data_test)
    filename_without_extension, _ = os.path.splitext(name_sae_dump)
    dataset.store_data(trackedObjects, config.path_store_data, trackManager.frame_rate, config.time_diff_prediction, name_dump=filename_without_extension)


if __name__ == "__main__":
    main()