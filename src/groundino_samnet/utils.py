import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, List
from torchvision.transforms import transforms
from groundingdino.datasets import transforms as T
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
from DataSets.getdata import PermuteTensor
from segment_anything1.utils.amg import remove_small_regions
import cv2

MODES = ["single", "batch"]
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

class PostProcessor:
    def __init__(self):
        pass

    def __call__(self):
        return self
    def purge_null_index(self, boxes: Union[torch.Tensor, List[torch.Tensor]], 
                          logits: Union[torch.Tensor, List[torch.Tensor]], 
                          phrases: Union[torch.Tensor, List[torch.Tensor]],
                          mode:str) -> Tuple[Union[torch.Tensor, 
                                                   List[torch.Tensor]], 
                                                   Union[torch.Tensor, List[torch.Tensor]], 
                                                                                   Union[List[str], List[List[str]]]]:
        """
            Purge null index from boxes, logits, and phrases.
        """

        if mode not in MODES:
            raise ValueError(f"Unrecognized prediction mode. Please select one of the allowed modes: {MODES}")

        if mode == "single":
            filtered_data = [(box, logit, phrase) for box, logit, phrase in zip(boxes, logits, phrases) if phrase]
            if not filtered_data:
                #raise ValueError("No valid data found. No phrases for batch.")
                return boxes,logits,phrases
            new_boxes, new_logits, new_phrases = zip(*filtered_data)
            new_boxes = torch.stack(new_boxes)
            new_logits = torch.stack(new_logits)

        elif mode == "batch":
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
                                     image_shape: Tuple,
                                     threshold: float,
                                     boxes: torch.Tensor, 
                                     logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Select non-overlapping boxes based on the IoU threshold.
        """
        W, H = image_shape
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
            overlap_indices = (iou_matrix[min_iou_index] > threshold).nonzero(as_tuple=True)[0].tolist()
            remaining_indices = [idx for idx in remaining_indices if idx not in overlap_indices]

        if len(selected_indices) > 2:
            selected_logits = logits[selected_indices]
            top_indices = torch.argsort(selected_logits, descending=True)[:2]
            selected_indices = torch.tensor(selected_indices)[top_indices].tolist()
        else:
            selected_indices = selected_indices[:2]

        return boxes[selected_indices], logits[selected_indices], selected_indices

    def postprocess_box(self,
                        image_shape: Tuple,
                        threshold: float,
                        boxes_list: Union[torch.Tensor, List[torch.Tensor]],
                        logits_list: Union[torch.Tensor, List[torch.Tensor]], 
                        phrases_list: Union[torch.Tensor, List[torch.Tensor]],
                        mode: str) -> Union[Tuple[torch.Tensor,torch.Tensor,List],Tuple[List[torch.Tensor],List[torch.Tensor],List[List]]]:
        """
        Process the boxes, logits, and phrases.
        """
        if mode not in MODES:
            raise ValueError(f"Unrecognized prediction mode. Please select one of the allowed modes: {MODES}")

        boxes_without_null, logits_without_null, phrases_without_null = self.purge_null_index(boxes=boxes_list,
                                                                                              logits=logits_list,
                                                                                              phrases=phrases_list,
                                                                                              mode=mode)

        if mode == "single":
            selected_boxes, selected_logits, selected_indices = self.select_non_overlapping_boxes(image_shape=image_shape,
                                                                                                  threshold=threshold,
                                                                                                  boxes=boxes_without_null, 
                                                                                                  logits=logits_without_null)
            new_phrases = [phrases_without_null[i] for i in selected_indices]
            return selected_boxes, selected_logits, new_phrases

        elif mode == "batch":
            new_boxes, new_logits, new_phrases = [], [], []
            for boxes, logits, phrases in zip(boxes_without_null, logits_without_null, phrases_without_null):
                selected_boxes, selected_logits, selected_indices = self.select_non_overlapping_boxes(boxes, 
                                                                                                      logits)
                selected_phrases = [phrases[i] for i in selected_indices]
                new_boxes.append(selected_boxes)
                new_logits.append(selected_logits)
                new_phrases.append(selected_phrases)

            return new_boxes, new_logits, new_phrases

    def postprocess_masks(self, masks: Union[torch.Tensor, List[torch.Tensor]], area_thresh: float) -> Union[torch.Tensor, List[torch.Tensor]]:
        def process_masks(mask_list: torch.Tensor, area_thresh: float, mode: str) -> torch.Tensor:
            """Apply remove_small_regions to a list of masks and return a stacked tensor."""
            masks_np = [remove_small_regions(mask.squeeze().detach().cpu().numpy(), area_thresh, mode)[0] for mask in mask_list]
            processed_masks = [np.expand_dims(mask, axis=0) for mask in masks_np]  # Add an extra dimension
            return torch.stack([torch.from_numpy(mask) for mask in processed_masks], dim=0)
        if isinstance(masks, list):
            processed_masks = []
            for mask_list in masks:
                masks_without_holes = process_masks(mask_list, area_thresh, "holes")
                masks_processed = process_masks(masks_without_holes, area_thresh, "islands")
                processed_masks.append(masks_processed)
            return processed_masks
        else:
            masks_without_holes = process_masks(masks, area_thresh, "holes")
            masks_processed = process_masks(masks_without_holes, area_thresh, "islands")
            return masks_processed

class PostProcessor2:
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self,):
        return self
    
    #Eliminar cajas se que sobre ponen por un umbral
    #Elimino cajas que se solapan con un umbral muy alto (casi identicas)
    def filter_boxes_by_iou(self,
                            boxes: torch.Tensor,
                            logits: torch.Tensor,
                            phrases: List, 
                            batch_mode: bool = False,
                            threshold: float = 0.85,**args) -> Union[Tuple[torch.Tensor, torch.Tensor, List], 
                                                                     Tuple[List[torch.Tensor],List[torch.Tensor],List[List]]]:
        """
        Elimina cajas que tienen un solapamiento mayor al umbral con cualquier otra caja.

        :param boxes: Tensor de cajas (N, 4) en formato xyxy
        :param iou_matrix: Matriz de solapamiento (N, N) obtenida con box_iou
        :param threshold: Umbral de solapamiento para eliminar cajas
        :return: Tensor filtrado de cajas
        """
        # Crear una máscara para las cajas que deben ser eliminadas
        verbose = args.get("verbose",False)
        def process_single(boxes: torch.Tensor,logits: torch.Tensor, phrases: List) -> Tuple[torch.Tensor, torch.Tensor, List]:
            nonlocal verbose
            H,W= self.shape
            boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            filtered_boxes = boxes_xyxy.clone()
            filtered_logits = logits.clone()
            filtered_phrases = phrases
            num_boxes = len(boxes_xyxy)
            while True:
                if num_boxes <= 1:
                    break

                # Calcular la matriz de solapamiento
                iou_matrix, _ = box_iou(filtered_boxes, filtered_boxes)
                iou_matrix.fill_diagonal_(0)  # Ignorar el solapamiento con la misma caja
                if verbose:
                    print(iou_matrix)
                num_boxes = iou_matrix.size(0)
                # Encontrar las cajas a eliminar
                to_remove_indices = set()
                for i in range(num_boxes):
                    for j in range(num_boxes):
                        if iou_matrix[i, j] > threshold:
                            to_remove_indices.add(i)
                            break  # Salir del bucle para evitar marcar múltiples cajas innecesariamente
                    if to_remove_indices:
                        break
                
                if not to_remove_indices:
                    break
                
                # Filtrar las cajas que no están en la lista de eliminación
                remaining_indices = list(set(range(num_boxes)) - to_remove_indices)
                filtered_boxes = filtered_boxes[remaining_indices]
                filtered_logits = filtered_logits[remaining_indices]
                filtered_phrases = [filtered_phrases[i] for i in remaining_indices]

            filtered_boxes = box_xyxy_to_cxcywh(filtered_boxes)  / torch.tensor([W, H, W, H])
            return filtered_boxes,filtered_logits,filtered_phrases

        if batch_mode:
            filtered_data = [process_single(boxes=box,logits=log,phrases=phrase) for box,log,phrase in zip(boxes,logits,phrases)]
            return zip(*filtered_data)
        else:
            return process_single(boxes=boxes,
                                  logits=logits,
                                  phrases=phrases)
    """
    #Hace lo mismo que delete_big_boxes (revisar)
    def delete_more_overlaped_boxes(self,
                                    boxes: torch.Tensor,
                                    logits: torch.Tensor,
                                    phrases: List,
                                    threshold: float= 0.2,
                                    batch_mode: bool = False,
                                    **args) -> Union[Tuple[torch.Tensor, torch.Tensor, List], 
                                                     Tuple[List[torch.Tensor],List[torch.Tensor],List[List]]]:
        
        verbose = args.get("verbose",False)
        def process_single(boxes: torch.Tensor, logits: torch.Tensor, phrases: List) -> Tuple[torch.Tensor, torch.Tensor, List]:
            nonlocal verbose
            H, W = self.shape
            boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            filtered_boxes = boxes_xyxy.clone()
            filtered_logits = logits.clone()
            filtered_phrases = phrases[:]
            
            while True:
                iou_matrix, _ = box_iou(filtered_boxes, filtered_boxes)
                iou_matrix.fill_diagonal_(0)
                if verbose:
                    print(iou_matrix)
                num_boxes = iou_matrix.size(0)
                to_remove_indices = set()

                for i, column in enumerate(iou_matrix):
                    count = (column > threshold).sum().item()
                    if count >= 2:
                        to_remove_indices.add(i)
                        break

                if not to_remove_indices:
                    break

                remaining_indices = list(set(range(num_boxes)) - to_remove_indices)
                filtered_boxes = filtered_boxes[remaining_indices]
                filtered_logits = filtered_logits[remaining_indices]
                filtered_phrases = [filtered_phrases[i] for i in remaining_indices]

            filtered_boxes_cxcywh = box_xyxy_to_cxcywh(filtered_boxes) / torch.Tensor([W, H, W, H])
            return filtered_boxes_cxcywh, filtered_logits, filtered_phrases

        if batch_mode:
            filtered_data = [process_single(box, log, phrase) for box, log, phrase in zip(boxes, logits, phrases)]
            return zip(*filtered_data)
        else:
            return process_single(boxes=boxes, 
                                  logits=logits, 
                                  phrases=phrases)
    """

    def iterative_box_removal(self,
                              boxes: torch.Tensor, 
                              logits: torch.Tensor, 
                              phrases: List, 
                              threshold: float = np.finfo(np.float32).eps, 
                              batch_mode: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, List], 
                                                                 Tuple[List[torch.Tensor],List[torch.Tensor],List[List]]]:
        #Elimina cajas que se solapan con varias
        def process_single(boxes: torch.Tensor, logits: torch.Tensor, phrases: List) -> Tuple[torch.Tensor, torch.Tensor, List]:
            H,W = self.shape
            boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            filtered_boxes = boxes_xyxy.clone()
            filtered_logits = logits.clone()
            filtered_phrases = phrases[:]

            while True:
                iou_matrix, _ = box_iou(filtered_boxes, filtered_boxes)
                iou_matrix.fill_diagonal_(0)
                overlaps_count = (iou_matrix > threshold).sum(dim=1)
                candidates = torch.where(overlaps_count >= 2)[0]

                if len(candidates) == 0:
                    break

                max_overlaps_idx = overlaps_count[candidates].argmax()
                box_to_remove = candidates[max_overlaps_idx].item()
                num_boxes = filtered_boxes.shape[0]
                remaining_indices = list(set(range(num_boxes)) - {box_to_remove})
                filtered_boxes = filtered_boxes[remaining_indices]
                filtered_logits = filtered_logits[remaining_indices]
                filtered_phrases = [filtered_phrases[i] for i in remaining_indices]

            filtered_boxes_cxcywh = box_xyxy_to_cxcywh(filtered_boxes) / torch.Tensor([W, H, W, H])
            return filtered_boxes_cxcywh, filtered_logits, filtered_phrases

        if batch_mode:
            filtered_results = [process_single(box, log, phrase) for box, log, phrase in zip(boxes, logits, phrases)]
            return zip(*filtered_results)
        else:
            return process_single(boxes, logits, phrases)

    """
    #Creo que esta es igual que la anterior (revisar)
    #busca en todas las columnas de la matrix si una caja esta solapada con dos cajas o mas, de ser asi posiblemente es una caja grande
    def delete_big_boxes(self,
                         boxes: torch.Tensor,
                         logits: torch.Tensor,
                         phrases: List,
                         batch_mode: bool = False,
                         **args) -> Union[Tuple[torch.Tensor, torch.Tensor, List], 
                                          Tuple[List[torch.Tensor],List[torch.Tensor],List[List]]]:
        
        verbose = args.get("verbose",False)
        def process_single(boxes,logits,phrases):
            nonlocal verbose
            H,W = self.shape
            boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            filtered_boxes = boxes_xyxy.clone()
            filtered_logits = logits.clone()
            filtered_phrases = phrases
            #busca en todas las columnas de la matrix si una caja esta solapada con dos cajas o mas, de ser asi posiblemente es una caja grande
            while True:
                iou_matrix, _ = box_iou(filtered_boxes, filtered_boxes)
                iou_matrix.fill_diagonal_(0)  # Ignorar el solapamiento con la misma caja
                if verbose:
                    print(iou_matrix)
                num_boxes = iou_matrix.size(0)
                to_remove_indices = set()
                for i,column in enumerate(iou_matrix): 
                    count = (column > 0.15).sum().item()
                    if count >= 2:
                        to_remove_indices.add(i)
                        break
                if not to_remove_indices:
                    break
            
                remaining_indices = list(set(range(num_boxes)) - to_remove_indices)
                filtered_boxes = filtered_boxes[remaining_indices]
                filtered_logits = filtered_logits[remaining_indices]
                filtered_phrases = [filtered_phrases[i] for i in remaining_indices]
                        
            filtered_boxes = box_xyxy_to_cxcywh(filtered_boxes)  / torch.tensor([W, H, W, H])
            return filtered_boxes, filtered_logits, filtered_phrases
        
        if batch_mode:
            filtered_data = [process_single(boxes=box,logits=log,phrases=phrase) for box,log,phrase in zip(boxes,logits,phrases)]
            return zip(*filtered_data)
        else: 
            return process_single(boxes=boxes,
                                  logits=logits,
                                  phrases=phrases)
    """
    
    #Purgar los indices nulos ahora los hace con los que solamente tengan un IoU menor a 0.20
    def purge_null_index(self,boxes: Union[torch.Tensor, List[torch.Tensor]], 
                        logits: Union[torch.Tensor, List[torch.Tensor]], 
                        phrases: Union[torch.Tensor, List[torch.Tensor]],
                        batch_mode: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, List], 
                                                           Tuple[List[torch.Tensor],List[torch.Tensor],List[List]]]:
        """
            Purge null index from boxes, logits, and phrases.
        """
        if batch_mode:
            null_indices = [
            {idx for idx, (phrase, logit) in enumerate(zip(phrases_batch, logits_batch)) 
            if phrase == '' and logit < 0.20}
            for logits_batch, phrases_batch in zip(logits, phrases)
            ] #Agregado
            new_boxes, new_logits, new_phrases = [], [], []

            for boxes_batch, logits_batch, phrases_batch, null_indices_batch in zip(boxes, logits, phrases, null_indices):
                if len(logits_batch) == len(null_indices_batch):
                    new_boxes.append(boxes_batch)
                    new_logits.append(logits_batch)
                    new_phrases.append(phrases_batch) #momentaneo
                elif not null_indices_batch:
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
        else:
            filtered_data = [(box, logit, phrase) for box, logit, phrase in zip(boxes, logits, phrases) if phrase]
            if not filtered_data:
                #raise ValueError("No valid data found. No phrases for batch.")
                return boxes,logits,phrases #momentaneo
            new_boxes, new_logits, new_phrases = zip(*filtered_data)
            new_boxes = torch.stack(new_boxes)
            new_logits = torch.stack(new_logits)

        return new_boxes, new_logits, new_phrases
    
    def postprocess_boxes(self,boxes,logits,phrases,batch_mode=False):
        if batch_mode:
            boxes1,logits1,phrases1 = self.purge_null_index(boxes=boxes,logits=logits,phrases=phrases,batch_mode=True)
            boxes2,logits2,phrases2 = self.filter_boxes_by_iou(boxes=boxes1,logits=logits1,phrases=phrases1,batch_mode=True)
            boxes3,logits3,phrases3 = self.iterative_box_removal(boxes=boxes2,logits=logits2,phrases=phrases2,batch_mode=True)
            boxes_f,logits_f,phrases_f = self.filter_boxes_by_iou(boxes=boxes3,logits=logits3,phrases=phrases3,threshold=np.finfo(np.float32).eps,batch_mode=True)
        else:
            boxes1,logits1,phrases1 = self.purge_null_index(boxes=boxes,logits=logits,phrases=phrases)
            boxes2,logits2,phrases2 = self.filter_boxes_by_iou(boxes=boxes1,logits=logits1,phrases=phrases1)
            boxes3,logits3,phrases3 = self.iterative_box_removal(boxes=boxes2,logits=logits2,phrases=phrases2)
            boxes_f,logits_f,phrases_f = self.filter_boxes_by_iou(boxes=boxes3,logits=logits3,phrases=phrases3,threshold=np.finfo(np.float32).eps)
        return boxes_f, logits_f, phrases_f
    
    def postprocess_masks(self, masks: Union[torch.Tensor, List[torch.Tensor]], area_thresh: float) -> Union[torch.Tensor, List[torch.Tensor]]:
        def process_masks(mask_list: torch.Tensor, area_thresh: float, mode: str) -> torch.Tensor:
            """Apply remove_small_regions to a list of masks and return a stacked tensor."""
            masks_np = [remove_small_regions(mask.squeeze().detach().cpu().numpy(), area_thresh, mode)[0] for mask in mask_list]
            processed_masks = [np.expand_dims(mask, axis=0) for mask in masks_np]  # Add an extra dimension
            return torch.stack([torch.from_numpy(mask) for mask in processed_masks], dim=0)
        if isinstance(masks, list):
            processed_masks = []
            for mask_list in masks:
                masks_without_holes = process_masks(mask_list, area_thresh, "holes")
                masks_processed = process_masks(masks_without_holes, area_thresh, "islands")
                processed_masks.append(masks_processed)
            return processed_masks
        else:
            masks_without_holes = process_masks(masks, area_thresh, "holes")
            masks_processed = process_masks(masks_without_holes, area_thresh, "islands")
            return masks_processed
        
if __name__ == "__main__":
    def test_postprocess_box():
        # Define las dimensiones
        N = 3
        boxes = torch.rand(N, 4) 
        logits = torch.rand(N)
        phrases = ["","",""]

        boxes,logits,phrases = PostProcessor().purge_null_index(boxes,logits,phrases,"single")
        print(boxes)
        print(logits)
        print(phrases)

    def test_postprocess_masks():
        # Create dummy data
        masks = torch.randint(0, 2, (3, 1, 5, 5), dtype=torch.bool).to(device='cuda')  # Tensor of shape (N, W, H, C)
        

        processor = PostProcessor()
        
        # Test with a single tensor
        processed_masks = processor.postprocess_masks(masks, area_thresh=500)
        print("Processed masks (single tensor):")
        print(processed_masks)

                # Test with a list of tensors
        mask_list = [torch.randint(0, 2, (5, 5, 3), dtype=torch.bool) for _ in range(3)]
        processed_mask_list = processor.postprocess_masks(mask_list, area_thresh=5)
        print("\nProcessed masks (list of tensors):")
        print(processed_mask_list)
        


    # Run the test
    #test_postprocess_masks()
    test_postprocess_box()

