import os
import urllib.request
from 

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

SAM_NAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth"
}


class SamLangDino():
    def __init__(self,):
        self.__file_path = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(self.__file_path,"weights")
        self.default_sam = "vit_h"
    def download_weights(self,
                         SAM:str,):
        os.makedirs(self.weights_path,exist_ok=True)
        if SAM not in SAM_NAMES.keys:
            print(f"Modelo de SAM no reconocible, utilizando por defecto <{self.default_sam}>") 
            SAM = self.default_sam
        

        


        
