import argparse
import os
import torch
from model.unet_model import UNet
from train import train


torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
unet = UNet()
use_CUDA = True


if __name__ == "__main__":

    parser.add_argument("-v", "--visual", action="store_true")
    parser.add_argument("-l", "--lr", type=float, default=1e-5)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch", type=int, default=40)
    parser.add_argument("-r", "--retrain", type=bool, default=False)

    args = parser.parse_args()

    if args.visual:
        from visual import show_pred_mask
        from loader import train_loader

        trl = train_loader(1, shuffle=True)
        img, msk = next(trl)
        unet.load_state_dict(torch.load("unet.pkl"))
        show_pred_mask(unet, img, msk)

    else:
        if use_CUDA:
            unet.cuda()
        try:
            train(
                unet,
                lr=args.lr,
                epochs=args.epochs,
                batch=args.batch,
                CUDA=use_CUDA,
                shuffle=True,
                retrain=args.retrain,
            )
        except KeyboardInterrupt as e:
            print(e)
            torch.save(unet.state_dict(), "unet.pkl")
            print("net Saved")
