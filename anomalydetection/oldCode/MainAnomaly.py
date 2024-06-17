import sys
sys.path.append("/home/hanna/workspaces/AETrajectories")

import logging
import PipelineConnector
import CheckLoop
from Config import AnomalyConfig


log = logging.getLogger(__name__)


def main():
    pathModelParameters = AnomalyConfig().path_to_model_json
    pipeline = PipelineConnector()
    detector = CheckLoop(pipeline, pathModelParameters)
    pipeline.start()
    detector.checkLoop()


if __name__ == "__main__":
    main()