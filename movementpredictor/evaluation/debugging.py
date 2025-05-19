from collections import defaultdict
import os
from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.evaluation.label_box import create_video
import json


evalconfig = EvalConfig()


path_label_box = os.path.join(evalconfig.path_label_box, evalconfig.camera)
redo_video = defaultdict()

for entry in os.listdir(path_label_box):
    full_path = os.path.join(path_label_box, entry)

    if os.path.isdir(full_path):
        path_label = os.path.join(full_path, "labeldata.json")
        with open(path_label, "r") as json_file:
            labeldata = json.load(json_file)

        if labeldata["label"] == 13:
            print(labeldata["obj_id"])

        #labeldata["time_interval"] = []

        #with open(path_label, "w", encoding="utf-8") as json_file:
         #   json.dump(labeldata, json_file, indent=4)

        video_found = any(
            file.lower().endswith(".mp4") for file in os.listdir(full_path)
        )

        if not video_found:
            print(f"Folder '{entry}' has no out.mp4")
            redo_video[labeldata["obj_id"]] = [labeldata["time_interval"][0], labeldata["time_interval"][1]]

if len(redo_video) != 0:
    create_video(redo_video, evalconfig.path_sae_dumps, path_label_box)




