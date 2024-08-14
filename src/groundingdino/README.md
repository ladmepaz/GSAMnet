<div align="center">
  <img src="../../.asset/groundingdino/grounding_dino_logo.png" width="30%">
</div>

# Créditos

Este proyecto incluye contenido derivado de [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file), que está bajo la Apache License 2.0. Puedes encontrar la documentación y detalles de la licencia en el archivo [LICENSE](src/groundingdino/LICENSE.txt).

# :sauropod: Grounding DINO 

**[IDEA-CVR, IDEA-Research](https://github.com/IDEA-Research)** 

[Shilong Liu](http://www.lsl.zone/), [Zhaoyang Zeng](https://scholar.google.com/citations?user=U_cvvUwAAAAJ&hl=zh-CN&oi=ao), [Tianhe Ren](https://rentainhe.github.io/), [Feng Li](https://scholar.google.com/citations?user=ybRe9GcAAAAJ&hl=zh-CN), [Hao Zhang](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=zh-CN), [Jie Yang](https://github.com/yangjie-cv), [Chunyuan Li](https://scholar.google.com/citations?user=Zd7WmXUAAAAJ&hl=zh-CN&oi=ao), [Jianwei Yang](https://jwyang.github.io/), [Hang Su](https://scholar.google.com/citations?hl=en&user=dxN1_X0AAAAJ&view_op=list_works&sortby=pubdate), [Jun Zhu](https://scholar.google.com/citations?hl=en&user=axsP38wAAAAJ), [Lei Zhang](https://www.leizhang.org/)<sup>:email:</sup>.


[[`Paper`](https://arxiv.org/abs/2303.05499)] [[`Demo`](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo)] [[`BibTex`](#black_nib-citation)]




## :bulb: Highlight

- **Open-Set Detection.** Detect **everything** with language!
- **High Performance.** COCO zero-shot **52.5 AP** (training without COCO data!). COCO fine-tune **63.0 AP**.
- **Flexible.** Collaboration with Stable Diffusion for Image Editting.



## :star: Explanations/Tips for Grounding DINO Inputs and Outputs
- Grounding DINO accepts an `(image, text)` pair as inputs.
- It outputs `900` (by default) object boxes. Each box has similarity scores across all input words. (as shown in Figures below.)
- We defaultly choose the boxes whose highest similarities are higher than a `box_threshold`.
- We extract the words whose similarities are higher than the `text_threshold` as predicted labels.
- If you want to obtain objects of specific phrases, like the `dogs` in the sentence `two dogs with a stick.`, you can select the boxes with highest text similarities with `dogs` as final outputs. 
- Note that each word can be split to **more than one** tokens with different tokenlizers. The number of words in a sentence may not equal to the number of text tokens.
- We suggest separating different category names with `.` for Grounding DINO.
![model_explain1](../../.asset/groundingdino/model_explan1.PNG)
![model_explain2](../../.asset/groundingdino/model_explan2.PNG)

## :medal_military: Results

<details open>
<summary><font size="4">
COCO Object Detection Results
</font></summary>
<img src="../../.asset/groundingdino//COCO.png" alt="COCO" width="100%">
</details>

<details open>
<summary><font size="4">
ODinW Object Detection Results
</font></summary>
<img src="../../.asset/groundingdino//ODinW.png" alt="ODinW" width="100%">
</details>

## :sauropod: Model: Grounding DINO

Includes: a text backbone, an image backbone, a feature enhancer, a language-guided query selection, and a cross-modality decoder.

![arch](../../.asset/groundingdino/arch.png)

## :black_nib: Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```




