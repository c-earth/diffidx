import torch.nn as nn
from savemodel.save import SaveModel

class Test(nn.Module):
    pass

tracker = SaveModel(Test, None, None, './utils', './models/test', None, None, None, None)
tracker.save()