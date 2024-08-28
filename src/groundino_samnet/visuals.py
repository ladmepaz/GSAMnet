import supervision as sv
from groundingdino.util.inference import annotate
def plot_grid(images,idss,boxes,logits,phrases,**args):
  imag_max = args.get("image_max",10) #cambiado, truncado de 20 a 10
  if len(images) > imag_max:
    print(f"Warning: The amount displayed will be truncated to 10. You can change this value using the image_max argument, but there is a risk of not displaying the images correctly.")
    images = images[:20]
    idss = idss[:20]
    boxes = boxes[:20]
    logits = logits[:20]
    phrases = phrases[:20]
  annotated_frames = []
  for i in range(len(boxes)):
    annotated_frame = annotate(image_source=images[i], boxes=boxes[i], logits=logits[i], phrases=phrases[i])
    annotated_frames.append(annotated_frame)
  sv.plot_images_grid(
    images=annotated_frames,
    grid_size=(8, int(len(annotated_frames) / 8)+1),
    size=(15, 15),
    titles=idss
  )

def plot_image(image,boxes,logits,phrases):
  annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
  sv.plot_image(annotated_frame, (16, 16))