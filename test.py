import argparse
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms  # type: ignore
import yaml
from torchvision.utils import save_image  # type: ignore

from dataset.floorplan_dataset_maps_functional_high_res import (
    FloorplanGraphDataset,
    floorplan_collate_fn,
)
from misc.utils import _init_input, draw_graph, draw_masks
from models.models import Generator

# from models.models_improved import Generator


parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_cpu",
    type=int,
    default=16,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./checkpoints/pretrained.pth",
    help="checkpoint path",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="./data/sample_list.txt",
    help="path to dataset list file",
)
parser.add_argument("--out", type=str, default="./dump", help="output folder")
parser.add_argument(
    "--save_format",
    "-f",
    choices=["png", "json", "yaml", "svg"],
    default="png",
    help="output format (png, json, yaml, svg)",
)
opt = parser.parse_args()
print(opt)

# Create output dir
os.makedirs(opt.out, exist_ok=True)

# Initialize generator and discriminator
model = Generator()
# Load checkpoint to the correct device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(opt.checkpoint, map_location=device)
model.load_state_dict(state_dict, strict=True)
# Move model to device and set to eval mode
model = model.to(device).eval()

# Initialize variables
if torch.cuda.is_available():
    model.cuda()

# initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(
    opt.data_path, transforms.Normalize(mean=[0.5], std=[0.5]), split="test"
)
fp_loader = torch.utils.data.DataLoader(
    fp_dataset_test,
    batch_size=opt.batch_size,
    shuffle=False,
    collate_fn=floorplan_collate_fn,
)


# run inference
def _infer(graph, model, prev_state=None):

    # configure input to the network
    z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
    # run inference model
    with torch.no_grad():
        masks = model(
            z.to(device),
            given_masks_in.to(device),
            given_nds.to(device),
            given_eds.to(device),
        )
        masks = masks.detach().cpu().numpy()
    return masks


def main():
    for i, sample in enumerate(fp_loader):

        # draw real graph and groundtruth
        mks, nds, eds, _, _ = sample
        real_nodes = np.where(nds.detach().cpu() == 1)[-1]
        graph = [nds, eds]
        true_graph_obj, graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
        graph_im.save("./{}/graph_{}.png".format(opt.out, i))  # save graph

        # add room types incrementally
        _types = sorted(list(set(real_nodes)))
        selected_types = [_types[: k + 1] for k in range(10)]
        os.makedirs("./{}/".format(opt.out), exist_ok=True)

        # initialize layout
        state = {"masks": None, "fixed_nodes": []}
        masks = _infer(graph, model, state)
        im0 = draw_masks(masks.copy(), real_nodes)
        im0 = torch.tensor(np.array(im0).transpose((2, 0, 1))) / 255.0
        # save_image(im0, './{}/fp_init_{}.png'.format(opt.out, i), nrow=1, normalize=False) # visualize init image

        # generate per room type
        for _iter, _types in enumerate(selected_types):
            _fixed_nds = (
                np.concatenate([np.where(real_nodes == _t)[0] for _t in _types])
                if len(_types) > 0
                else np.array([])
            )
            state = {"masks": masks, "fixed_nodes": _fixed_nds}
            masks = _infer(graph, model, state)

        # save final floorplans
        imk = draw_masks(masks.copy(), real_nodes)
        imk = torch.tensor(np.array(imk).transpose((2, 0, 1))) / 255.0
        if opt.save_format == "png":
            save_image(imk, f"{opt.out}/fp_final_{i}.png", nrow=1, normalize=False)
        elif opt.save_format in ("json", "yaml"):
            data = {"nodes": real_nodes.tolist(), "masks": masks.tolist()}
            out_file = os.path.join(opt.out, f"fp_final_{i}.{opt.save_format}")
            with open(out_file, "w") as f:
                if opt.save_format == "json":
                    json.dump(data, f, indent=2)
                else:
                    yaml.dump(data, f, default_flow_style=False)
        elif opt.save_format == "svg":
            h, w = masks.shape[1], masks.shape[2]
            svg_lines = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            ]
            for nd, mask in zip(real_nodes, masks):
                binary = (mask > 0).astype("uint8") * 255
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    pts = cnt.squeeze()
                    if pts.ndim != 2:
                        continue
                    points_str = " ".join(f"{x},{y}" for x, y in pts.tolist())
                    svg_lines.append(
                        f'<polygon points="{points_str}" fill="none" stroke="black" id="room_{nd}" />'
                    )
            svg_lines.append("</svg>")
            out_file = os.path.join(opt.out, f"fp_final_{i}.svg")
            with open(out_file, "w") as f:
                f.write("\n".join(svg_lines))


if __name__ == "__main__":
    main()
