import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, List
from torchvision.transforms import transforms
from groundingdino.datasets import transforms as T
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, box_iou
from DataSets.Mamitas_Thermal_Dataset.Mamitas_Dataset import PermuteTensor

def load_image_from_PIL(img:Image.Image) -> torch.Tensor:
    """
        Load a PIL image while ensuring it meets the specifications required by GroundingDINO.

        Args:
            img: A single image PIL
        
        Returns:
            image: A single torch.Tensor for GroundingDINO
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(img,None)
    return image

def build_model(args):
    # we use register to maintain models from catdet6 on.
    from groundingdino.models import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)
    return model

def load_image(image: Union[Image.Image,
                            torch.Tensor,
                            np.ndarray]) -> torch.Tensor:
    """
        Convert images from various formats (PIL, torch.Tensor, np.ndarray) to a torch.Tensor to ensure compatibility with GroundingDINO.

        Args:
            image: A single image in various formats

        Returns:
            transformed_image: A single torch.Tensor with the transformed image
    """
    if isinstance(image, Image.Image):
        transformed_image = load_image_from_PIL(image)

    elif isinstance(image, torch.Tensor):
        if image.shape[0] != 3:
            image = image.permute((2, 0, 1))
        transformed_image = transforms.ToPILImage()(image)
        transformed_image = load_image_from_PIL(transformed_image)

    elif isinstance(image, np.ndarray):
        if image.shape[0] == 3:        
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                PermuteTensor((2, 0, 1)),
                transforms.ToPILImage(),
            ])  
        transformed_image = transform(image)
        transformed_image = load_image_from_PIL(transformed_image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Please provide a PIL Image, torch.Tensor, or np.ndarray.")

    return transformed_image

def convert_image_to_numpy(image: Union[Image.Image,
                                        torch.Tensor,
                                        np.ndarray]) -> np.ndarray:
    """
        Convert an image from various formats (PIL, Tensor) to a Numpy
        
        Args:
            image: The input image.

        Returns:
            The converted numpy array.
    """
    if isinstance(image,torch.Tensor):
        if image.shape[0] == 3:
            image = image.permute((1,2,0))
        image_array = image.numpy()
    elif isinstance(image, Image.Image):
        image_array = np.asarray(image)
    elif isinstance(image,np.ndarray):
        if image.shape[0] == 3:
            image = np.transpose(image,(1,2,0))
        image_array = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Please provide a PIL Image, torch.Tensor, or np.ndarray.")
    return image_array
import torch
from typing import List, Tuple, Union

class PostProcessor:
    def __init__(self, image_shape: tuple, threshold: float, mode: str):
        self.threshold = threshold
        self.mode = mode
        self.image_shape = image_shape
        self.MODES = ["single", "batch"]
        if self.mode not in self.MODES:
            raise ValueError(f"Unrecognized prediction mode. Please select one of the allowed modes: {self.MODES}")

    def purge_null_index(self, boxes: Union[torch.Tensor, List[torch.Tensor]], 
                          logits: Union[torch.Tensor, List[torch.Tensor]], 
                          phrases: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], 
                                                                                   Union[torch.Tensor, List[torch.Tensor]], 
                                                                                   Union[List[str], List[List[str]]]]:
        """
        Purge null index from boxes, logits, and phrases.
        """
        if self.mode == "single":
            filtered_data = [(box, logit, phrase) for box, logit, phrase in zip(boxes, logits, phrases) if phrase]
            if not filtered_data:
                raise ValueError("No valid data found. No phrases for batch.")
            new_boxes, new_logits, new_phrases = zip(*filtered_data)
            new_boxes = torch.stack(new_boxes)
            new_logits = torch.stack(new_logits)

        elif self.mode == "batch":
            null_indices = [
                {idx for idx, x in enumerate(phrases_batch) if x == ''}
                for phrases_batch in phrases
            ]
            new_boxes, new_logits, new_phrases = [], [], []

            for boxes_batch, logits_batch, phrases_batch, null_indices_batch in zip(boxes, logits, phrases, null_indices):
                if len(logits_batch) == len(null_indices_batch):
                    raise ValueError("No valid data found. No phrases for batch.")
                if not null_indices_batch:
                    new_boxes.append(boxes_batch)
                    new_logits.append(logits_batch)
                    new_phrases.append(phrases_batch)
                else:
                    filtered_boxes = [box for idx, box in enumerate(boxes_batch) if idx not in null_indices_batch]
                    filtered_logits = [logit for idx, logit in enumerate(logits_batch) if idx not in null_indices_batch]
                    filtered_phrases = [phrase for idx, phrase in enumerate(phrases_batch) if idx not in null_indices_batch]
                    new_boxes.append(torch.stack(filtered_boxes))
                    new_logits.append(torch.stack(filtered_logits))
                    new_phrases.append(filtered_phrases)

        return new_boxes, new_logits, new_phrases

    def select_non_overlapping_boxes(self,
                                     boxes: torch.Tensor, 
                                     logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Select non-overlapping boxes based on the IoU threshold.
        """
        W, H = self.image_shape
        boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        iou_matrix, _ = box_iou(boxes_xyxy, boxes_xyxy)
        iou_matrix.fill_diagonal_(0)

        selected_indices = []
        remaining_indices = list(range(len(boxes)))

        while remaining_indices:
            if len(selected_indices) >= 2:
                break

            max_iou_per_box = iou_matrix[remaining_indices, :][:, remaining_indices].max(dim=1).values
            min_iou_index = remaining_indices[max_iou_per_box.argmin().item()]
            selected_indices.append(min_iou_index)

            remaining_indices.remove(min_iou_index)
            overlap_indices = (iou_matrix[min_iou_index] > self.threshold).nonzero(as_tuple=True)[0].tolist()
            remaining_indices = [idx for idx in remaining_indices if idx not in overlap_indices]

        if len(selected_indices) > 2:
            selected_logits = logits[selected_indices]
            top_indices = torch.argsort(selected_logits, descending=True)[:2]
            selected_indices = torch.tensor(selected_indices)[top_indices].tolist()
        else:
            selected_indices = selected_indices[:2]

        return boxes[selected_indices], logits[selected_indices], selected_indices

    def postprocess_box(self,
                    boxes_list: Union[torch.Tensor, List[torch.Tensor]], 
                    logits_list: Union[torch.Tensor, List[torch.Tensor]], 
                    phrases_list: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], 
                                                                                        Union[torch.Tensor, List[torch.Tensor]], 
                                                                                        Union[List[str], List[List[str]]]]:
        """
        Process the boxes, logits, and phrases.
        """
        boxes_without_null, logits_without_null, phrases_without_null = self.purge_null_index(
            boxes=boxes_list,
            logits=logits_list,
            phrases=phrases_list
        )

        if self.mode == "single":
            selected_boxes, selected_logits, selected_indices = self.select_non_overlapping_boxes(boxes_without_null, logits_without_null)
            new_phrases = [phrases_without_null[i] for i in selected_indices]
            return selected_boxes, selected_logits, new_phrases

        elif self.mode == "batch":
            new_boxes, new_logits, new_phrases = [], [], []
            for boxes, logits, phrases in zip(boxes_without_null, logits_without_null, phrases_without_null):
                selected_boxes, selected_logits, selected_indices = self.select_non_overlapping_boxes(boxes, logits)
                selected_phrases = [phrases[i] for i in selected_indices]
                new_boxes.append(selected_boxes)
                new_logits.append(selected_logits)
                new_phrases.append(selected_phrases)

            return new_boxes, new_logits, new_phrases

def process_masks(masks):
    pass

if __name__ == "__main__":
    import random
    B = 3
    boxes1 = torch.randn(B, 4)
    boxes2 = torch.randn(B, 4)
    logits1 = torch.randn(B)
    logits2 = torch.randn(B)
    phrases1 = ["f","f","f"]
    phrases2 = ["f","f","f"]

    boxes = [boxes1, boxes2]
    logits = [logits1, logits2]
    phrases = [phrases1, phrases2]

    boxes, logits, phrases = process_multiple_sets((480,680),boxes,logits,phrases,0.1,"batch")
    print(boxes)