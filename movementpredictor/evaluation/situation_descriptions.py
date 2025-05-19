import json
from PIL import Image
import os
import logging
from movementpredictor.evaluation.eval_config import EvalConfig
from pg_videollava import PGVideoLLavaForCausalLM, PGVideoLLavaProcessor


log = logging.getLogger(__name__)
evalconfig = EvalConfig()


def load_video_datasample(video_path):
    frames_path = os.path.join(video_path, "frames")
    with open(os.path.join(video_path, "bboxes.json")) as f:
        anno = json.load(f)

    fnames = sorted(anno.keys())
    frames = []
    boxes  = []
    for fn in fnames:
        img = Image.open(os.path.join(frames_path, fn)).convert("RGB")
        frames.append(img)
        boxes.append(anno[fn])  # [x1, y1, x2, y2]

    return frames, boxes


path_label_box = os.path.join(evalconfig.path_label_box, evalconfig.camera)
additional_info = []
frames_per_video = []
bboxs_per_video = []

for i, entry in enumerate(os.listdir(path_label_box)):
    #if i > 10: 
     #   break
    video_path = os.path.join(path_label_box, entry)

    with open(os.path.join(video_path, "labeldata.json"), "r") as json_file:
        labeldata = json.load(json_file)
    
    if labeldata["label"] != 12:        # try for all "too slow" movements
        continue
    
    frames, boxes = load_video_datasample(video_path)
    additional_info.append(labeldata)
    frames_per_video.append(frames)
    bboxs_per_video.append(boxes)


processor = PGVideoLLavaProcessor.from_pretrained("mbzuai/pg-videollava")
model     = PGVideoLLavaForCausalLM.from_pretrained("mbzuai/pg-videollava").to("cuda")

task_prompt = (
    "Describe what happens in the marked region, "
    "then classify the behavior into one of: very fast, very slow, almost crash, u-turn, none of these.\n"
    "Output format: Description: <your description>  Category: <one of the four>."
)

inputs = processor(
    frames_per_video,
    boxes=bboxs_per_video,
    task="describe",         
    prompt=task_prompt, 
    return_tensors="pt",
    padding=True 
).to("cuda")

output_ids = model.generate(
    **inputs, 
    max_new_tokens=100,       
    num_beams=4,              
    early_stopping=True       
)

for seq_ids, info in zip(output_ids, additional_info):
    desc = processor.decode(seq_ids, skip_special_tokens=True)
    print("----\n", info["obj_id"], "\n", desc)