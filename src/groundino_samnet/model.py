import os
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'  #Permtute usar las funciones especiales de SAM2 como el manejo eficiente de memoria y los bloques de atencion

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional 
from huggingface_hub import hf_hub_download
from segment_anything1.build_sam import sam_model_registry
from segment_anything1.predictor import SamPredictor
from segment_anything1.config import SAM1_MODELS, SAM_NAMES_MODELS
from segment_anything2.config import SAM2_MODELS
from segment_anything2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util import box_ops
from groundingdino.util.inference import predict, load_model
from torchvision.ops import box_convert
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
from DataSets.getdata import thermal_feet_dataset, ToBoolTensor, PermuteTensor
from torchvision.transforms import transforms
import torch.nn as nn
class GSamnet(nn.Module):
    def __init__(self,dino_args =None,sam_args= None):
        super(GSamnet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dino_args = dino_args
        self.sam_args = sam_args

        self.transform_mask = transforms.Compose([
            ToBoolTensor(),
            PermuteTensor((1,2,0)) # (WxHxC)
        ])
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.uint8),
            PermuteTensor((1,2,0)) #(WxHxC)
        ])

    def dummy_input(self):
        try:
            image,_,_ = thermal_feet_dataset.load_instance(root_name_img=os.path.join(os.path.dirname(__file__),"dummy_input","t0.jpg"),
                                                            root_name_mask=os.path.join(os.path.dirname(__file__),"dummy_input","t0.png"),
                                                            merge_image=True,
                                                            transform_img=self.transform_img,
                                                            transform_mask=self.transform_mask,
                                                            dataset="mtf")
            
            self.forward([image])
            print("Notice: Warm-up inference completed.")
        except Exception as e:
            print(f"Error occurred loading the dummy image for warm-up inference: {e}")
            
    def forward(self,image,return_all= False):
        unbatch = False
        if (isinstance(image,torch.Tensor) and image.shape[0] == 1) or (isinstance(image,list) and len(image) == 1):
            unbatch = True
        
    
        text_prompt = self.dino_args.get("text_prompt","feet . foot . soles")
        box_threshold = self.dino_args.get("box_threshold",0.10)
        text_threshold = self.dino_args.get("text_threshold",0.30)
        box_process_threshold = self.dino_args.get("box_process_threshold",0.10)
        postproccesingv2 = self.dino_args.get("postprocessing",True)

        area_threshold = self.sam_args.get("area_threshold",700)
        boxes, logits, phrases, shape = self.predict_dino_batch(model=self.dino_args["model"],
                                                        images=image,
                                                        text_prompt=text_prompt,
                                                        box_threshold=box_threshold,
                                                        text_threshold=text_threshold,
                                                        box_process_threshold=box_process_threshold,
                                                        postproccesingv2=postproccesingv2)
        H,W = shape
        if self.sam_args["points"]:
            boxes_p = [box_cxcywh_to_xyxy(box) * torch.tensor([W,H,W,H]) for box in boxes]
            result = [box_xyxy_to_point(box) for box in boxes_p]
            points_coords,points_labels = zip(*result)
        else:
            points_coords = None
            points_labels = None

        if self.sam_args["model"].name == "SAM1":
            mask = self.predict_SAM1_batch(model=self.sam_args["model"],
                                    images=image,
                                    area_thresh=area_threshold,
                                    boxes=boxes,
                                    points_coords=points_coords,
                                    points_labels=points_labels)
                                    
            
        elif self.sam_args["model"].name == "SAM2":
            boxes = [box_convert(box * torch.Tensor([W, H, W, H]), in_fmt="cxcywh", out_fmt="xyxy") for box in boxes]

            mask = self.predict_SAM2_batch(model=self.sam_args["model"],
                                    images=image,
                                    area_thresh=area_threshold,
                                    boxes=boxes,
                                    points_coords=points_coords,
                                    points_labels=points_labels)
        
        
        if self.sam_args["torch"]:
            mask = [torch.Tensor(maski) for maski in mask]
        if return_all:
            if unbatch:
                return boxes[0],logits[0],phrases[0],mask[0]
            else:
                return boxes,logits,phrases,mask
        else:
            if unbatch:
                return mask[0]
            else:
                return mask

        
    def predict_dino(self, 
                     model,
                     image: Union[Image.Image, 
                                  torch.Tensor, 
                                  np.ndarray], 
                     text_prompt: str, 
                     box_threshold: float, 
                     text_threshold: float,
                     box_process_threshold: float,
                     **args) -> torch.Tensor:
        """
            Run the Grounding DINO model for bounding box prediction.

            Args:
                image: The input image with (WxHxC) shape.
                text_prompt: The text prompt for bounding box prediction.
                box_threshold: The threshold for bounding box prediction.
                text_threshold: The threshold for text prediction.
                UnNormalize (optional): Whether to unnormalize the image. Defaults to False.

            Returns:
                The predicted bounding boxes with (B,4) shape with logits and phrases.
        """
        postproccesingv1 = args.get("postproccesingv1",False)
        postproccesingv2 = args.get("postproccesingv2",False)

        image_trans = load_image(image)
        image_array = convert_image_to_numpy(image)
        shape =  image_array.shape[:2]
        boxes, logits, phrases = predict(model=model,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        
        if postproccesingv1:
            boxes,logits,phrases = PostProcessor().postprocess_box(image_shape=shape,
                                                            threshold=box_process_threshold,
                                                            boxes_list=boxes,
                                                            logits_list=logits,
                                                            phrases_list=phrases,
                                                            mode="single")
        if postproccesingv2:
            boxes,logits,phrases = PostProcessor2(shape=shape).postprocess_boxes(boxes,logits,phrases)

        return boxes, logits, phrases, shape
    def predict_dino_batch(self,
                           model,
                           images:List[Union[Image.Image,torch.Tensor,np.ndarray]],
                           text_prompt: str, 
                           box_threshold: float, 
                           text_threshold: float,
                           box_process_threshold: float,
                           **args) -> Tuple[List[torch.Tensor],List[torch.Tensor],List[torch.Tensor]]:
        """
            Run the Grounding DINO model for batch prediction.

            Args:
                images: The input images with (WxHxC) shape.
                text_prompt: The text prompt for bounding box prediction.
                box_threshold: The threshold for bounding box prediction.
                text_threshold: The threshold for text prediction.
                Normalize (optional): Whether to normalize the image. Defaults to False

            Returns:
                The predicted bounding boxes with (B,4) shape with logits and phrases.
        """
        postproccesingv1 = args.get("postproccesingv1",False)
        postproccesingv2 = args.get("postproccesingv2",True)
        results = list(map(lambda image: self.predict_dino(model=model,
                                                           image=image,
                                                           text_prompt=text_prompt,
                                                           box_threshold=box_threshold,
                                                           text_threshold=text_threshold,
                                                           box_process_threshold=box_process_threshold), images))
        boxes, logits, phrases,_ = zip(*results)
        boxes = list(boxes)
        logits = list(logits)
        phrases = list(phrases)
        shape =  images[0].shape[:2]
        if postproccesingv1:
            boxes,logits,phrases = PostProcessor().postprocess_box(image_shape=shape,
                                                                   threshold=box_process_threshold,
                                                                   boxes_list=boxes,
                                                                   logits_list=logits,
                                                                   phrases_list=phrases,
                                                                   mode="batch")
        if postproccesingv2:
            boxes,logits,phrases = PostProcessor2(shape=shape).postprocess_boxes(boxes,logits,phrases,batch_mode=True)
        return boxes, logits, phrases,shape
    
    def predict_SAM1(self,
                     model,
                     image: Union[Image.Image, 
                                 torch.Tensor,
                                 np.ndarray], 
                     area_thresh: float,
                     boxes: Optional[torch.Tensor] = None,
                     points_coords: Optional[torch.Tensor] = None,
                     points_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
            Run the SAM1 model for image segmentation.

            Args:
                image: The input image with (WxHxC) shape.
                area_thresh: Minimum mask area
                boxes: The bounding boxes for segmentation. Defaults to None.
                points_coords: The coordinates of the points for segmentation. Defaults to None.
                points_labels: The labels of the points for segmentation. Defaults to None.

            Returns
                The predicted segmentation mask with (WxHx1) shape.
    """
        image_array = convert_image_to_numpy(image)
        transformed_boxes,transformed_points,points_labels = self.__prep_prompts_SAM1(boxes,
                                                                                 points_coords,
                                                                                 points_labels,
                                                                                 image_array.shape[:2])

        model.set_image(image_array)
        masks, _, _ = model.predict_torch(point_coords=transformed_points.to(self.device) if transformed_points is not None else None,
                                          point_labels=points_labels.to(self.device) if points_labels is not None else None,
                                          boxes=transformed_boxes.to(self.device) if transformed_boxes is not None else None,
                                          multimask_output=False,)
        model.reset_image()
        masks = PostProcessor().postprocess_masks(masks=masks,
                                                area_thresh=area_thresh)
        masks = masks.cpu()
        mask = torch.any(masks,dim=0).permute(1,2,0).numpy()
        return mask
    
    
    def predict_SAM1_batch(self,
                           model,
                           images:List[Union[Image.Image,
                                             torch.Tensor,
                                             np.ndarray]],
                           area_thresh: float,
                           boxes:Optional[List[torch.Tensor]] = None,
                           points_coords:Optional[List[torch.Tensor]] = None,
                           points_labels:Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
            Run the SAM1 model for batch prediction.

            Args:
                images: The input images with (WxHxC) shape.
                boxes: List of bounding boxes for each image. Can be None.
                points_coords: List of point coordinates for each image. Can be None.
                points_labels: List of point labels for each image. Can be None.

            Returns:
                The predicted masks for each image.
        """
        if points_coords is not None and points_labels is None:
            raise ValueError("If 'points_coords' is provided, 'points_labels' must also be provided, and vice versa.")
        elif points_labels is not None and points_coords is None:
            raise ValueError("If 'points_labels' is provided, 'points_coords' must also be provided, and vice versa.")
        
        if boxes is None:
            boxes = [None] * len(images)
        if points_coords is None:
            points_coords = [None] * len(images)
            points_labels = [None] * len(images)

        if not (len(images) == len(boxes) == len(points_coords) == len(points_labels)):
            raise ValueError("The lengths of 'images', 'boxes', 'points_coords', and 'points_labels' must match.")
        
        def process_image(image: Union[Image.Image,torch.Tensor,np.ndarray],
                          box: Optional[torch.Tensor],
                          point_coord: Optional[torch.Tensor],
                          point_label: Optional[torch.Tensor]) -> torch.Tensor:
            """
                Process a single image with its corresponding boxes and points.

                Args:
                    image: The input image with (WxHxC) shape.
                    box: The bounding boxes for the image.
                    point_coords: The point coordinates for the image.
                    point_labels: The point labels for the image.

                Returns:
                    np.ndarray: The predicted mask for the image.
            """
            nonlocal model
            mask = self.predict_SAM1(model=model,
                                     image=image,
                                     area_thresh=area_thresh,
                                     boxes=box,
                                     points_coords=point_coord,
                                     points_labels=point_label)
            return mask
        results = [process_image(image, box, point_coords, point_labels) for image, box, point_coords, point_labels in zip(images, boxes, points_coords, points_labels)]
        return results
    
    def predict_SAM2(self,
                     model,
                     image: Union[Image.Image,
                                  torch.Tensor,
                                  np.ndarray],
                     area_thresh: float,
                     boxes: np.ndarray, 
                     point_coords: np.ndarray,
                     point_labels: np.ndarray) -> torch.Tensor:
        
        """
            Run the SAM2 model for image segmentation.

            Args:
                image: The input image with (WxHxC) shape.
                area_thresh: Minimum mask area
                boxes: The bounding boxes for segmentation. Defaults to None.
                points_coords: The coordinates of the points for segmentation. Defaults to None.
                points_labels: The labels of the points for segmentation. Defaults to None.

            Returns
                The predicted segmentation mask with (WxHx1) shape.
            
        """
        image_array = convert_image_to_numpy(image)
        box,point_coords,point_labels = self.__prep_prompts_SAM2(boxes=boxes,points_coords=point_coords,points_labels=point_labels)
        with torch.inference_mode(),  torch.autocast("cuda", dtype=torch.bfloat16):
            model.set_image(image_array)
            masks,_,_ = model.predict(point_coords=point_coords,
                              point_labels=point_labels,
                              box=box,
                              multimask_output=False)

            model.reset_predictor()
            masks = torch.Tensor(masks).to(torch.bool)
            masks = PostProcessor().postprocess_masks(masks=masks,
                                                area_thresh=area_thresh)
            masks = masks.cpu()
            mask = torch.any(masks,dim=0).permute(1,2,0).numpy()
        return mask
    
    def predict_SAM2_batch(self,
                           model,
                           images: List[Union[Image.Image,torch.Tensor,np.ndarray]],
                           points_coords: List[Union[np.ndarray]],
                           points_labels: List[Union[np.ndarray]],
                           boxes: List[Union[np.ndarray]],
                           area_thresh: float,
                           multimask_output: bool = False) -> List[torch.Tensor]:
        """
            Run the SAM2 model for batch prediction.

            Args:
                images: List of the input images with (WxHxC) shape.
                boxes: List of bounding boxes for each image. Can be None.
                points_coords: List of point coordinates for each image. Can be None.
                points_labels: List of point labels for each image. Can be None.

            Returns:
                The predicted masks for each image.
        """
        masks = []
        for image,box,point,label in zip(images,boxes,points_coords,points_labels):
            
            mask = self.predict_SAM2(model=model,
                                        image=image,
                                        area_thresh=area_thresh,
                                        boxes=box,
                                        point_coords=point,
                                        point_labels=label)
            masks.append(mask)
        return masks

    def __prep_prompts_SAM1(self,
                       boxes: Optional[torch.Tensor],
                       points_coords: Optional[torch.Tensor],
                       points_labels: Optional[torch.tensor],
                       dims: tuple) -> Tuple[Optional[torch.Tensor],Optional[torch.Tensor],Optional[torch.Tensor]]:
        """
            Prepare the prompts to be used by SAM1.

            Args: 
                boxes: Tensor of box coordinates
                points_coords: Tensor of point coordinates
                points_labels: Tensor of point labels
                dims: Image dimensions

            Return:
                The processed prompts
        """

        H,W = dims 

        if boxes is not None:
            clip_valor = np.clip(boxes[0][0], 0, 1)
            if clip_valor == boxes[0][0]:
                boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W,H,W,H])
                transformed_boxes = self.SAM1.transform.apply_boxes_torch(boxes, (W,H))
            else:
                transformed_boxes = boxes #Agregado 
        else:
            transformed_boxes=None

        if points_coords is not None and points_labels is None:
            raise ValueError("If 'points_coords' is provided, 'points_labels' must also be provided, and vice versa.")
        elif points_labels is not None and points_coords is None:
            raise ValueError("If 'points_labels' is provided, 'points_coords' must also be provided, and vice versa.")
        elif points_coords is not None and points_labels is not None:
            transformed_points = self.SAM1.transform.apply_coords_torch(points_coords, dims)
        else:
             transformed_points = None
             points_labels = None
    
        return transformed_boxes,transformed_points,points_labels
    
        
    def __prep_prompts_SAM2(self,
                       boxes: Optional[torch.Tensor],
                       points_coords: Optional[torch.Tensor],
                       points_labels: Optional[torch.tensor]) -> Tuple[Optional[torch.Tensor],Optional[torch.Tensor],Optional[torch.Tensor]]:
        if boxes is not None:
            transformed_boxes = np.asarray(boxes)
        else:
            transformed_boxes = None

        if points_coords is not None:
            transformed_points = np.asarray(points_coords)
            transformed_labels = np.asarray(points_labels)
        else: 
            transformed_points,transformed_labels = (None,None)

        return transformed_boxes,transformed_points,transformed_labels
    
    def __Build_GroundingDINO(self,return_model:Optional[bool] =  None):
        """
            Build the Grounding DINO model.
        """
        repo_id = "ShilongLiu/GroundingDINO"
        filename = "groundingdino_swint_ogc.pth"
        cache_config = "GroundingDINO_SwinT_OGC.cfg.py"
        try:
            cache_config_file = hf_hub_download(repo_id=repo_id, filename=cache_config)
            pth_file = hf_hub_download(repo_id=repo_id, filename=filename)
        except:
            raise RuntimeError(f"Error downloading GroundingDINO model. Please ensure that the {repo_id}/{cache_config} file exists in huggingface_hub and the {filename} checkpoint is functional.")
        
        try:
            groundingdino = load_model(cache_config_file,pth_file, device=self.device)
            if return_model is not None:
                if return_model:
                    return groundingdino

        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the GroundingDINO model: {str(e)}")


    
    def __Build_SAM1(self,
                     SAM:str,
                     return_model:Optional[bool] = None) -> None:
        """
            Build the SAM1 model.

            Args:
                SAM: The name of the SAM model to build.
        """
        try:
            checkpoint_url = SAM1_MODELS[SAM]
            sam = sam_model_registry[SAM]()
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
        except Exception as e:
            raise RuntimeError(f"Error downloading SAM1. Please ensure the model is correct: {SAM} and that the checkpoint is functional: {checkpoint_url}.")
        try:
            sam.load_state_dict(state_dict, strict=True)
            sam.to(device=self.device)
            SAM1 = SamPredictor(sam)
            if return_model is not None:
                if return_model:
                    return SAM1
        except Exception as e:
            raise RuntimeError(f"SAM1 model can't be compile: {str(e)}")

    def __Build_SAM2(self,
                     SAM:str,
                     return_model: Optional[bool] = None) -> None:
        """
            Build the SAM2 model.

            Args:
                SAM: The name of the SAM model to build.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint_url = SAM2_MODELS[SAM]
            SAM2 = SAM2ImagePredictor.from_pretrained(checkpoint_url,device=device)
            if return_model is not None:
                if return_model:
                    return SAM2
        except Exception as e:
            raise RuntimeError(f"Error downloading or Compile {SAM} model. Please ensure that {SAM2_MODELS[SAM]} is functional: {e}")

def load_models(model):

    if model == "dino":
        modelo = GSamnet()._GSamnet__Build_GroundingDINO(return_model=True)
        modelo.name = "dino"
        return modelo
    else:
        if model in SAM1_MODELS:
            modelo = GSamnet()._GSamnet__Build_SAM1(model,return_model=True)
            modelo.name = "SAM1"
            return modelo
        elif model in SAM2_MODELS:
            modelo = GSamnet()._GSamnet__Build_SAM2(model, return_model=True)
            modelo.name = "SAM2"
            return modelo
        else:
            raise ValueError(f"{model} is not a model available try {SAM1_MODELS} for SAM1 models or {SAM2_MODELS} for SAM2 models")    
        

if __name__ == "__main__":
    from groundino_samnet.utils import PostProcessor, PostProcessor2, load_image, convert_image_to_numpy, box_xyxy_to_point
    sam = load_models("sam2_t")
    dino = load_models("dino")

    sam_args = {
        "model": sam,
        "area_threshold": 700,
        "points": True,
        "torch": True
    }

    dino_args = {
    "model": dino,
    "text_prompt": "feet . foot . soles",
    "box_threshold": 0.10,
    "text_threshold": 0.30,
    "box_process_threshold": 0.10,
    "postprocessing": True
    }

    model = GSamnet(dino_args=dino_args,sam_args=sam_args)

else:
    from .utils import PostProcessor, PostProcessor2, load_image, convert_image_to_numpy, box_xyxy_to_point