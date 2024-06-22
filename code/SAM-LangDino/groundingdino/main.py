import os
import torch
import numpy as np
from PIL import Image
import urllib.request
from Segment_Anything.build_sam import sam_model_registry
from Segment_Anything.predictor import SamPredictor
from huggingface_hub import hf_hub_download
from util.utils import (SLConfig, 
                        build_model, 
                        clean_state_dict, 
                        change_instance_image_dino, 
                        change_instace_image_sam)

from util.inference import predict, load_image_from_PIL
from util import box_ops,point_ops
from torchvision.transforms import transforms
from typing import Union, Optional

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Build_GroundingDINO(self,):

        repo_id = "ShilongLiu/GroundingDINO"
        filename = "groundingdino_swinb_cogcoor.pth"
        pth_config = "GroundingDINO_SwinB.cfg.py"
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=pth_config)
        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        args.device = self.device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print(f"Model loaded from {cache_file} \n => {log}")
        model.eval()

        self.groundingdino = model
    
    def Build_Sam(self,
                  SAM:str,):
        os.makedirs(self.weights_path,exist_ok=True)
        if SAM not in SAM_NAMES.keys:
            print(f"Modelo de SAM no reconocible, utilizando por defecto <{self.default_sam}>") 
            SAM = self.default_sam
        checkpoint_url = SAM_MODELS[self.sam_type]
        try:
            sam = sam_model_registry[self.sam_type]()
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
            sam.load_state_dict(state_dict, strict=True)
        except:
            raise ValueError(f"Problemas al descargar sam, asegurese que el modo es correcto: {self.sam_type} \
                    y funcione el checkpoint: {checkpoint_url}.")
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

    def predict_dino(self, 
                     image: Union[Image.Image, 
                                  torch.Tensor, 
                                  np.ndarray], 
                     text_prompt: str, 
                     box_threshold: float, 
                     text_threshold: float) -> torch.Tensor:
        
        image_trans = change_instance_image_dino(image)
        image_array = change_instace_image_sam(image)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         remove_combined=self.return_prompts,
                                         device=self.device)
        W, H = image_array.shape[0],image_array.shape[1]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        return boxes, logits, phrases
    
    def predict_sam_with_boxes(self,
                               image: Union[Image.Image, 
                                            torch.Tensor,
                                            np.ndarray], 
                                boxes: torch.Tensor) -> torch.Tensor:
        
        image_array = change_instace_image_sam(image)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        self.sam.reset_image()
        return masks.cpu()
    
    def predict_sam_with_points(self,
                           image,
                           boxes: Optional[torch.Tensor] = None,
                           points_coords: Optional[torch.Tensor] = None,
                           points_labels: Optional[torch.Tensor] = None,
                           neg_points: Optional[bool] = False) -> torch.Tensor:
        image_array = change_instace_image_sam(image)
        W,H = image_array.shape[0],image_array.shape[1]
        if boxes is not None:
            boxesxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W,H,W,H])
            points_coords, points_labels =  point_ops.box_xyxy_to_points(boxesxy, neg_point=neg_points)

        self.sam.set_image(image_array)
        if points_coords is not None or points_labels is not None:

            transformed_points = self.sam.transform.apply_coords_torch(points_coords, image_array.shape[:2])
            masks, _, _ = self.sam.predict_torch(
                point_coords=transformed_points.to(self.sam.device),
                point_labels=points_labels.to(self.sam.device),
                boxes=None,
                multimask_output=False,
            )
        else:
            raise ValueError("points_coords and points_labels must be different to None if boxes is None")
        
        self.sam.reset_image()
        return masks.cpu()
    def predict_sam_with_boxes_points(self,
                                 image: Union[Image.Image, torch.Tensor],
                                 boxes: Optional[torch.Tensor] = None,
                                 points_coords: Optional[torch.Tensor] = None,
                                 points_labels: Optional[torch.Tensor] = None,
                                 neg_points: Optional[bool] = False) -> torch.Tensor:
        if boxes is None:
            raise ValueError("Boxes must be provided")
        
        if (points_coords is not None or points_labels is not None) and (points_coords is None or points_labels is None):
            raise ValueError("If points_coords or points_labels are provided, both must be provided")
        
        image_array = change_instace_image_sam(image)
        H, W = image_array.shape[:2]
        
        if points_coords is None and points_labels is None:
            boxesxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H])
            points_coords, points_labels = point_ops.box_xyxy_to_points(boxesxy, neg_point=neg_points)
        
        self.sam.set_image(image_array)
        transformed_points = self.sam.transform.apply_coords_torch(points_coords, image_array.shape[:2])
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])

        masks, _, _ = self.sam.predict_torch(
            point_coords=transformed_points.to(self.sam.device),
            point_labels=points_labels.to(self.sam.device),
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        self.sam.reset_image()
        return masks.cpu()

        
