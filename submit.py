import time

import numpy as np
import torch

from loader import test_loader
from model.unet_model import UNet


torch.cuda.set_device(1)
unet = UNet()
unet.load_state_dict(torch.load("unet.pkl"))
unet = unet.cuda()


def rle_encode(preds):
    result = []
    pixels = preds.flatten(1)
    pixels[:, 0] = pixels[:, -1] = 0
    for pixel in pixels:
        runs = np.where(pixel[1:] != pixel[:-1])[0] + 2
        runs[1::2] = runs[1::2] - runs[:-1:2]
        result.append(" ".join(map(str, runs)))
    return result


if __name__ == "__main__":

    print("Start writing...")

    with open("predict.csv", "w") as f:
        f.write("id,rle_mask\n")

        total_num = 18000
        batch_size = 36
        times = total_num / batch_size
        print("{:.1f} times.".format(times))

        tl = test_loader(batch_size)
        for i, (names, imgs) in enumerate(tl):
            tic = time.time()

            imgs = imgs.cuda()
            preds = unet(imgs).view(batch_size, 101, 101)
            preds = preds.permute(0, 2, 1).contiguous()
            preds = preds.view(batch_size, -1) > 0.5
            codes = rle_encode(preds)
            torch.cuda.empty_cache()

            for ii, name in enumerate(names):
                f.write("{},{}\n".format(name, codes[ii]))

            delt = (time.time() - tic) / 60
            rest_min = delt * (times - i - 1)
            print(
                "epoch {} finished. Cost: {:.2f} min. Rest time: {:.2f} min".format(
                    i + 1, delt, rest_min
                )
            )
