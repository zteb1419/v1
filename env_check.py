import sys

import matplotlib
import numpy as np
import pandas as pd
import torch

print("interpreter", sys.version.split()[0])
print("environment", sys.prefix)
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("matplotlib", matplotlib.__version__)
print("torch", torch.__version__)

