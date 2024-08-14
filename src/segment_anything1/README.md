# Créditos

Este proyecto incluye contenido derivado de [Segment-Anythin-Model_1](https://github.com/facebookresearch/segment-anything/tree/main), que está bajo la Apache License 2.0. Puedes encontrar la documentación y detalles de la licencia en el archivo [LICENSE](LICENSE.txt).


# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](../../.asset/sam1/model_diagram.png?raw=true)

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

<p float="left">
  <img src="../../.asset/sam1/masks1.png?raw=true" width="37.25%" />
  <img src="../../.asset/sam1/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Dataset

See [here](https://ai.facebook.com/datasets/segment-anything/) for an overview of the datastet. The dataset can be downloaded [here](https://ai.facebook.com/datasets/segment-anything-downloads/). By downloading the datasets you agree that you have read and accepted the terms of the SA-1B Dataset Research License.

We save masks per image as a json file. It can be loaded as a dictionary in python in the below format.

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
```

Image ids can be found in sa_images_ids.txt which can be downloaded using the above [link](https://ai.facebook.com/datasets/segment-anything-downloads/) as well.

## Citing Segment Anything

If you use SAM or SA-1B in your research, please use the following BibTeX entry.

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
