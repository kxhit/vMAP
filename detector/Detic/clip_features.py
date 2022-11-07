import json
import torch
import clip

lvis_class = json.load(open('./datasets/metadata/lvis_v1_train_cat_info.json', 'r'))
lvis_list = [c['name'] for c in lvis_class]

clip_model, _ = clip.load("ViT-L/14", device='cpu')
with torch.no_grad():
    word_embed = clip.tokenize(lvis_list).to('cpu')
    word_features = clip_model.encode_text(word_embed)

lvis_dict = {lvis_list[i]: word_features[i] for i in range(len(lvis_list))}

torch.save(lvis_dict, "lvis_clip_dict.pt")