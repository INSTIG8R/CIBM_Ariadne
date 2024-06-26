from torch import tensor
from torchmetrics.classification import Dice
preds  = tensor([8, 9, 224, 224])
target = tensor([8, 1, 224, 224])
dice = Dice(average='micro')
dice(preds, target)

print(dice)