import supervision as sv
from groundingdino.util.inference import annotate
import matplotlib.pyplot as plt
import numpy as np
from .utils import convert_image_to_numpy
from typing import List, Tuple, Optional, Union
import torch

def plot_grid_dino(images,idss,boxes,logits,phrases,**args):
  imag_max = args.get("image_max",10) #cambiado, truncado de 20 a 10
  padding = args.get("padding",1)
  if len(images) > imag_max:
    print(f"Warning: The amount displayed will be truncated to {imag_max}. You can change this value using the image_max argument, but there is a risk of not displaying the images correctly.")
    images = images[:imag_max]
    idss = idss[:imag_max]
    boxes = boxes[:imag_max]
    logits = logits[:imag_max]
    phrases = phrases[:imag_max]
    images = [convert_image_to_numpy(image) for image in images]
  annotated_frames = []
  for i in range(len(boxes)):
    annotated_frame = annotate(image_source=images[i], boxes=boxes[i], logits=logits[i], phrases=phrases[i])
    annotated_frames.append(annotated_frame)
  plot_image_grid(
    images=annotated_frames,
    image_size=(20, 20),
    titles=idss,
    padding=padding
  )

def plot_image_grid(images: List, grid_size:Optional[Tuple] = None, image_size: Tuple = (10, 10), titles: List[str] = None, padding:Union[float,int] =1):
    """
    Plots a grid of images using matplotlib with padding between images.
    
    Args:
      images: List of images (numpy arrays or image-like objects).
      grid_size: Tuple (rows, cols) specifying the grid dimensions. If None, it is calculated automatically.
      image_size: Tuple (width, height) specifying the size of each image in the grid.
      titles: List of titles for each image. If None, no titles are displayed.
      padding: Padding size between images.
    """
    # Ensure images is a list
    if not isinstance(images, list):
        raise TypeError("Images should be provided as a list. For plot a individual image use 'plot_image'")
    
    # Determine grid size
    num_images = len(images)
    if grid_size is None:
        cols = 5
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size
    
    # Create a figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * (image_size[0] + padding), rows * (image_size[1] + padding)))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Plot each image
    for ax, img, title in zip(axes, images, titles or [None] * num_images):
        ax.imshow(img)
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=120)
    
    # Turn off axes for any unused subplots
    for ax in axes[num_images:]:
        ax.axis('off')
    
    # Adjust layout with padding
    plt.subplots_adjust(wspace=padding / image_size[0], hspace=padding / image_size[1])
    plt.show()

def plot_image(image,boxes,logits,phrases):
  convert_image_to_numpy(image)
  annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
  sv.plot_image(annotated_frame, (16, 16))


#Code from SAM2
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_result(image,mask=None, point_coords=None, box_coords=None, input_labels=None, borders=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if mask is not None:
        if isinstance(mask,torch.Tensor):
            mask = np.asarray(mask)
        show_mask(mask, plt.gca(), borders=borders)

    if point_coords is not None:
        assert input_labels is not None

        if isinstance(point_coords,torch.Tensor):
            point_coords = np.asarray(point_coords)
        if isinstance(input_labels,torch.Tensor):
            point_coords = np.asarray(point_coords)
        show_points(point_coords, input_labels, plt.gca())
    if box_coords is not None:
        if isinstance(box_coords,torch.Tensor):
            box_coords = np.asarray(box_coords)
        for box in box_coords:
            show_box(box, plt.gca())
    plt.axis('off')
    plt.show()

#Arreglar
def plot_grid_sam(images,idss,boxes,points,labels,masks,**args):
  imag_max = args.get("image_max",10) #cambiado, truncado de 20 a 10
  padding = args.get("padding",1)
  if len(images) > imag_max:
    print(f"Warning: The amount displayed will be truncated to {imag_max}. You can change this value using the image_max argument, but there is a risk of not displaying the images correctly.")
    images = images[:imag_max]
    idss = idss[:imag_max]
    boxes = boxes[:imag_max]
    logits = logits[:imag_max]
    phrases = phrases[:imag_max]
    images = [convert_image_to_numpy(image) for image in images]
  annotated_frames = []
  for i in range(len(boxes)):
    annotated_frame = annotate(image_source=images[i], boxes=boxes[i], logits=logits[i], phrases=phrases[i])
    annotated_frames.append(annotated_frame)
  plot_image_grid(
    images=annotated_frames,
    image_size=(20, 20),
    titles=idss,
    padding=padding
  )