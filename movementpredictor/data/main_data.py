from movementpredictor.data.datamanagement import getTrackedBaseData, get_background_frame
from movementpredictor.data.datafilterer import DataFilterer
from movementpredictor.config import ModelConfig
from movementpredictor.data import dataset

import os
import logging

log = logging.getLogger(__name__)
config = ModelConfig()


def main():

    background_frame = get_background_frame(config.path_sae_data, config.dim_x, config.dim_y)
    dataset.store_frame(background_frame, config.path_store_data, config.path_model)

    for i in range(12):
        trackedObjects = getTrackedBaseData(config.path_sae_data, i)           # 1'220'175 it in total
        trackedObjects = DataFilterer().apply_filtering(trackedObjects) 
        dataset.store_data(trackedObjects, config.path_store_data, i)
            
    train_ds = dataset.getTorchDataSet(os.path.join(config.path_store_data, "clustering"))
    train_dl = dataset.getTorchDataLoader(train_ds, shuffle=False)

    dataset.plotDataSamples(train_dl, 50)


if __name__ == "__main__":
    main()
