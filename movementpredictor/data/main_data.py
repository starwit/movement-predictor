from movementpredictor.data.datamanagement import getTrackedBaseData
from movementpredictor.data.datafilterer import DataFilterer
from movementpredictor.config import ModelConfig
from movementpredictor.data import dataset

import gc
import logging

log = logging.getLogger(__name__)
config = ModelConfig()


def main():

    for i in range(12):
        frames_dict, trackedObjects = getTrackedBaseData(config.path_sae_data, config.dim_x, config.dim_y, i)           # 1'220'175 it in total
        trackedObjects = DataFilterer().apply_filtering(trackedObjects) 

        dataset.makeTorchDataSet(frames_dict, trackedObjects, config.path_store_data, i)
        frames_dict, trackedObjects = None, None
        trackedObjects = None
        gc.collect()

    train_ds = dataset.getTorchDataSet(config.path_store_data, "train_cnn", 0)
    train_dl = dataset.getTorchDataLoader(train_ds)

    dataset.plotDataSamples(train_dl, 20)


if __name__ == "__main__":
    main()
