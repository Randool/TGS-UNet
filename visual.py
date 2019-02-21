import torch
import numpy as np
import matplotlib.pyplot as plt


def show_pred_mask(net, img=None, msk=None):
    pred = net(img) > 0.5
    pred = np.array(pred.view(101, 101))
    msks = np.hstack([pred, np.array(msk.view(101, 101))])
    plt.imshow(msks)
    plt.title("predicted | true mask")
    plt.show()
