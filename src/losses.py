import torch.nn as nn


def get_loss_function():

    return nn.CrossEntropyLoss()