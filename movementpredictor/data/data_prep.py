from movementpredictor.data.datafilterer import DataFilterer
from movementpredictor.config import ModelConfig
from movementpredictor.data.dataset import store_data, getTorchDataSet, getTorchDataLoader, plotDataSamples
from movementpredictor.data.datamanagement import get_background_frame, store_frame, TrackingDataManager, load_background_frame

import os
import logging

log = logging.getLogger(__name__)
config = ModelConfig()


def build_train_dataset():

    name_sae_dump =  os.path.basename(config.path_sae_data_train)
    filename_without_extension, _ = os.path.splitext(name_sae_dump)

    trackManager = TrackingDataManager()
    trackedObjects = trackManager.getTrackedBaseData(config.path_sae_data_train, inferencing=False)
    trackedObjects = DataFilterer().apply_filtering(trackedObjects)
    store_data(trackedObjects, config.class_of_interest, config.path_store_data, config.time_diff_prediction, folder="train", 
               frame_rate=config.frame_rate, name_dump=filename_without_extension)

    if config.visualize:
        # background_fame - only for visualization, it is not needed for the model 
        background_frame = get_background_frame(config.path_sae_data_train, config.pixel_per_axis)
        store_frame(background_frame, config.path_store_data)

        train_ds = getTorchDataSet(os.path.join(config.path_store_data, "train"), pixel_per_axis=config.pixel_per_axis)
        train_dl = getTorchDataLoader(train_ds, shuffle=False)
        frame = load_background_frame(config.path_store_data)
        plotDataSamples(train_dl, 50, config.path_plots, frame)


def build_test_dataset():

    trackManager = TrackingDataManager()

    for path_sae_data_test in config.paths_sae_data_test:

        log.info(f"Processing test data from: {path_sae_data_test}")
        trackedObjects = trackManager.getTrackedBaseData(path_sae_data_test, inferencing=True)
        trackedObjects = DataFilterer().apply_filtering(trackedObjects)

        name_sae_dump =  os.path.basename(path_sae_data_test)
        filename_without_extension, _ = os.path.splitext(name_sae_dump)
        
        store_data(trackedObjects, config.class_of_interest, config.path_store_data, config.time_diff_prediction, folder="test",
                frame_rate=config.frame_rate, name_dump=filename_without_extension)