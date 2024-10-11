import glob
import pickle
import cv2
import os
import torch
from tqdm import tqdm

from wav2lip.models import Wav2Lip


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} for inference.".format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	tmp_model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	tmp_model.load_state_dict(new_s)

	tmp_model = tmp_model.to(device)
	return tmp_model.eval()



def read_imgs(img_list):
    frames = []
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


class avatar_model_class:

    def __init__(self, opt):
        self.opt = opt

        gol_modelavatar_id = opt.avatar_id
        gol_modelavatar_path = f"./data/avatars/{gol_modelavatar_id}"
        gol_modelfull_imgs_path = f"{gol_modelavatar_path}/full_imgs"
        gol_modelface_imgs_path = f"{gol_modelavatar_path}/face_imgs"
        gol_modelcoords_path = f"{gol_modelavatar_path}/coords.pkl"

        with open(gol_modelcoords_path, "rb") as f:
            self.gol_coord_list_cycle = pickle.load(f)

        input_img_list = glob.glob(
            os.path.join(gol_modelfull_imgs_path, "*.[jpJP][pnPN]*[gG]")
        )
        input_img_list = sorted(
            input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.gol_frame_list_cycle = read_imgs(input_img_list)

        input_face_list = glob.glob(
            os.path.join(gol_modelface_imgs_path, "*.[jpJP][pnPN]*[gG]")
        )
        input_face_list = sorted(
            input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.gol_face_list_cycle = read_imgs(input_face_list)

        self.gol_model = load_model("./models/wav2lip.pth")
