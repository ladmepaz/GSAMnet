from typing import Union
from PIL import Image
from groundingdino.util.inference import load_image
from torchvision.transforms import transforms
from groundingdino.datasets import transforms as T
import numpy as np
import torch
from DataSets.Mamitas_Thermal_Dataset.Mamitas_Dataset import PermuteTensor

def load_image_from_PIL(img):
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

def load_trans_image(image: Union[Image.Image,torch.Tensor,np.ndarray]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image_trans = load_image_from_PIL(image)
    elif isinstance(image, torch.Tensor):
        if image.shape[0] != 3:
            image = image.permute((2,0,1))
        image_trans = transforms.ToPILImage()(image)
        image_trans = load_image_from_PIL(image_trans)
    elif isinstance(image, np.ndarray):
        if image.shape[0] == 3:        
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                PermuteTensor((2,0,1)),
                transforms.ToPILImage(),
            ])  
        image_trans = transform(image)
        image_trans = load_image_from_PIL(image_trans)
    return image_trans

def change_image_instance(image: Union[Image.Image,torch.Tensor]) -> torch.Tensor:
    if isinstance(image,torch.Tensor):
        if image.shape[0] == 3:
            image = image.permute((1,2,0))
        image_array = image.numpy()
    elif isinstance(image, Image.Image):
        image_array = np.asarray(image)
    else:
        if image.shape[0] == 3:
            image = np.transpose(image,(1,2,0))
        image_array = image
    return image_array

def purge_null_index(boxes: torch.Tensor, 
                     logits: torch.Tensor, 
                     phrases: torch.Tensor):
  new_boxes = []
  new_logits = []
  new_phrases = []
  for i in range(len(boxes)):
    null_index = [i for i, x in enumerate(phrases[i]) if x == '']
    new_boxes.append(eliminated_elemets(boxes[i],null_index))
    new_logits.append(eliminated_elemets(logits[i],null_index))
    new_phrases.append(eliminated_elemets(phrases[i],null_index))
  return new_boxes, new_logits, new_phrases

def eliminated_elemets(array,indexs):
  return [item for idx, item in enumerate(array) if idx not in indexs]

def select_non_overlapping_boxes(image_shape: tuple, 
                                 boxes: torch.Tensor, 
                                 logits:torch.Tensor, 
                                 threshold:float):
    W,H = image_shape
    boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W,H,W,H])

    iou_matrix,_ = box_iou(boxes_xyxy, boxes_xyxy)

    # Poner a cero la diagonal para evitar solapamiento consigo mismo
    iou_matrix.fill_diagonal_(0)

    selected_indices = []
    remaining_indices = list(range(len(boxes)))

    while remaining_indices:
        if len(selected_indices) >= 2:
            break

        # Seleccionar el índice de la caja con el menor solapamiento máximo
        max_iou_per_box = iou_matrix[remaining_indices, :][:, remaining_indices].max(dim=1).values
        min_iou_index = remaining_indices[max_iou_per_box.argmin().item()]
        selected_indices.append(min_iou_index)

        # Eliminar las cajas seleccionadas de los índices restantes
        remaining_indices.remove(min_iou_index)

        # Eliminar cajas que se solapen con la caja seleccionada
        overlap_indices = (iou_matrix[min_iou_index] > threshold).nonzero(as_tuple=True)[0].tolist()
        for idx in overlap_indices:
            if idx in remaining_indices:
                remaining_indices.remove(idx)

    # Si se seleccionaron más de dos cajas, quedarse con las dos cajas con los logits más altos
    if len(selected_indices) > 2:
        selected_logits = logits[selected_indices]
        top_indices = torch.argsort(selected_logits, descending=True)[:2]
        selected_indices = torch.tensor(selected_indices)[top_indices].tolist()
    else:
        selected_indices = selected_indices[:2]

    return boxes[selected_indices], logits[selected_indices], selected_indices

def process_multiple_sets(image_shape: tuple, 
                          boxes_list: torch.Tensor, 
                          logits_list: torch.Tensor, 
                          phrases_list: torch.Tensor, 
                          threshold: float):
    new_boxes = []
    new_logits = []
    new_phrases = []
    for boxes, logits, phrases in zip(boxes_list, logits_list, phrases_list):
        selected_boxes, selected_logits, selected_indices = select_non_overlapping_boxes(image_shape, boxes, logits, threshold)

        # Seleccionar las frases correspondientes a las cajas seleccionadas
        selected_phrases = [phrases[i] for i in selected_indices]
        new_boxes.append(selected_boxes)
        new_logits.append(selected_logits)
        new_phrases.append(selected_phrases)
    return new_boxes, new_logits, new_phrases

def process_box_batch(shape: tuple,
                      boxes: torch.Tensor,
                      logits: torch.Tensor,
                      phrases: torch.Tensor,
                      box_threshold: float):
    boxes_without_null, logits_without_null, phrases_without_null = purge_null_index(boxes=boxes,
                                                                                    logits=logits,
                                                                                    phrases=phrases)
        
    for i in range(len(logits_without_null)):
        boxes_without_null[i] = torch.stack(boxes_without_null[i])
        logits_without_null[i] = torch.stack(logits_without_null[i])

    new_boxes, new_logits, new_phrases = process_multiple_sets(shape,
                                                               boxes_without_null, 
                                                               logits_without_null, 
                                                               phrases_without_null, 
                                                               threshold=box_threshold)
    
    return new_boxes, new_logits, new_phrases