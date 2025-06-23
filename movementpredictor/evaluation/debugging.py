from collections import Counter, defaultdict
import glob
import os
import shutil
from matplotlib import pyplot as plt
from movementpredictor.evaluation.eval_config import EvalConfig
from movementpredictor.evaluation.label_box import create_video
import json


evalconfig = EvalConfig()


#path_label_box_with_labels = os.path.join(evalconfig.path_label_box, "RangelineSMedicalDr_1-3-5-10")
#list_ids_with_labels = os.listdir(path_label_box_with_labels)

path_label_box = os.path.join(evalconfig.path_label_box, evalconfig.camera)
redo_video = defaultdict()
all_event_labels = []
all_lost_event_labels = []
videos_found = 0

for entry in os.listdir(path_label_box):

    #if entry in list_ids_with_labels:
     #   full_path_label = os.path.join(path_label_box_with_labels, entry)

#        if os.path.isdir(full_path_label):
 #           path_label = os.path.join(full_path_label, "labeldata.json")

  #          with open(path_label, "r") as json_file:
   #             labeldata = json.load(json_file)
            
    #        label = labeldata["label"]
    
    #else:
     #   continue

    full_path = os.path.join(path_label_box, entry)

    if os.path.isdir(full_path):
        path_label = os.path.join(full_path, "labeldata.json")

        with open(path_label, "r") as json_file:
            labeldata = json.load(json_file)

        #labeldata["label"] = label

        #with open(path_label, "w", encoding="utf-8") as json_file:
         #   json.dump(labeldata, json_file, indent=4)

        if labeldata["label"] == "None":
            print(labeldata["obj_id"])

        #if labeldata["time_interval"][1] - labeldata["time_interval"][0] > 5000 and (labeldata["label"] == 0 or labeldata["label"] == 2):
         #   print(labeldata["obj_id"])

        #video_found = any(
         #   file.lower().endswith(".mp4") for file in os.listdir(full_path)
        #)

        #if video_found:
         #   videos_found += 1
        if labeldata["label"] != 0:
            all_event_labels.append(labeldata["label"])
        #labeldata["time_interval"] = []

        #with open(path_label, "w", encoding="utf-8") as json_file:
         #   json.dump(labeldata, json_file, indent=4)

        #for filepath in glob.glob(os.path.join(full_path, '*.mp4')):
         #   os.remove(filepath)

        #if not video_found: #and labeldata["label"] != 0: #and labeldata["label"] != -1:
         #   shutil.rmtree(full_path)
            #print(labeldata["label"], " at ", labeldata["obj_id"])
            #all_lost_event_labels.append(labeldata["label"])

         #   print(f"Folder '{entry}' has no out.mp4")
          #  redo_video[labeldata["obj_id"]] = [labeldata["time_interval"][0], labeldata["time_interval"][1]]

'''
label_counts = Counter(all_lost_event_labels)
sorted_labels = sorted(label_counts.keys())
frequencies = [label_counts[label] for label in sorted_labels]

plt.figure(figsize=(8, 5))
plt.bar(sorted_labels, frequencies)
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.title("Histogram of lost event labels")
plt.xticks(sorted_labels)  
plt.grid(axis='y')
#plt.ylim(0, 94)
plt.tight_layout()
path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
os.makedirs(path, exist_ok=True)
plt.savefig(os.path.join(path, "LostEventFrequencies.png"))
plt.close()
'''

label_counts = Counter(all_event_labels)
sorted_labels = sorted(label_counts.keys())
frequencies = [label_counts[label] for label in sorted_labels]

plt.figure(figsize=(8, 5))
plt.bar(sorted_labels, frequencies)
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.title("Histogram of event labels")
plt.xticks(sorted_labels)  
plt.grid(axis='y')
plt.ylim(0, 94)
plt.tight_layout()
path = os.path.join("movementpredictor/evaluation/plots", evalconfig.camera)
os.makedirs(path, exist_ok=True)
plt.savefig(os.path.join(path, "FoundEventFrequencies.png"))
plt.close()


#if len(redo_video) != 0:
 #   create_video(redo_video, evalconfig.path_sae_dumps, path_label_box)



