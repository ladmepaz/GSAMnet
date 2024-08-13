import numpy as np

valor  = 2
valor_clip = np.clip(valor,0,1)
if valor == valor_clip:
    print("si")