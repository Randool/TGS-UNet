import time

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Function

from loader import train_loader


class DiceCoeff(nn.Module):
    """ 单类 IoU loss function """

    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, inputs, targets):
        B = targets.size(0)

        inputs_flat = inputs.view(B, -1)
        targets_flat = targets.view(B, -1)

        interscetion = (inputs_flat * targets_flat).sum(1) + 0.01
        union = inputs_flat.sum(1) + targets_flat.sum(1) + 0.01

        IoU = 2.0 * interscetion / union
        loss = 1 - IoU.sum() / B

        return loss


def train(unet, lr, epochs, batch, CUDA, shuffle, retrain):
    if not retrain:
        print("Loading model...")
        unet.load_state_dict(torch.load("unet.pkl"))
        print("OK!")
    print("=============\nStart training\nlr: {}\nepoch: {}".format(lr, epochs))
    print(
        "batch: {}\nshuffle: {}\nCUDA: {}\n=============".format(batch, shuffle, CUDA)
    )

    optimizer = optim.SGD(unet.parameters(), lr=lr, momentum=0.7)
    criterion = nn.BCELoss()  # change loss function

    tic = time.time()
    Tic = tic
    for epoch in range(epochs):
        epoch_loss = 0
        tloader = train_loader(batch, shuffle)

        for i, D in enumerate(tloader):
            optimizer.zero_grad()
            img, msk = D
            if CUDA:
                img = img.cuda()
                msk = msk.cuda()
            pred_msk = unet(img).view(-1)

            # contiguous的作用是将浅拷贝变成深拷贝，以此使用view
            loss = criterion(pred_msk, msk.contiguous().view(-1))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        toc = time.time()
        print(
            "epoch {} finished --- loss: {:.6f} --- Cost: {:.2f} min".format(
                epoch, epoch_loss / i, (toc - tic) / 60
            )
        )
        tic = toc

    torch.save(unet.state_dict(), "unet.pkl")
    print("Finished. Cost: {:.2f} min".format((tic - Tic) / 60))
