from movementpredictor.data.datafilterer import DataFilterer
from movementpredictor.config import ModelConfig
from movementpredictor.data import dataset, datamanagement

import os
import logging

log = logging.getLogger(__name__)
config = ModelConfig()


def main():
    # get background_fame only for visualization - it is not needed for the model 
    background_frame = datamanagement.get_background_frame(config.path_sae_data_train, config.pixel_per_axis)
    datamanagement.store_frame(background_frame, config.path_store_data)

    name_sae_dump =  os.path.basename(config.path_sae_data_train)
    filename_without_extension, _ = os.path.splitext(name_sae_dump)

    trackManager = datamanagement.TrackingDataManager()
    trackedObjects = trackManager.getTrackedBaseData(config.path_sae_data_train, inferencing=False)
    trackedObjects = DataFilterer().apply_filtering(trackedObjects)
    dataset.store_data(trackedObjects, config.path_store_data, config.time_diff_prediction, "train", frame_rate=10, name_dump=filename_without_extension)

    # visualization
    train_ds = dataset.getTorchDataSet(os.path.join(config.path_store_data, "train"), pixel_per_axis=config.pixel_per_axis)
    train_dl = dataset.getTorchDataLoader(train_ds, shuffle=False)
    frame = datamanagement.load_background_frame(config.path_store_data)
    dataset.plotDataSamples(train_dl, 50, config.path_plots, frame)

if __name__ == "__main__":
    main()
