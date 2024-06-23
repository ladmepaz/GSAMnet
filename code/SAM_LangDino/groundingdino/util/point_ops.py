import torch 
def box_xyxy_to_points(boxes: torch.Tensor,
                      neg_point:bool=False):
  """
    Args:

  """
  num_points = 5 if neg_point else 1
  num_box = len(boxes)
  points_coords = torch.zeros((num_box * num_points,2))
  points_labels = torch.ones(num_box * num_points)
  idx = 0
  for box in range(num_box):
    x = (boxes[box][0] + boxes[box][2]) / 2
    y = (boxes[box][1] + boxes[box][3]) / 2
    points_coords[box] = [x,y]
    if neg_point:
      for idx in range(num_points - num_box):
        points_coords[1] = [boxes[box][0], boxes[box][1]]
        points_coords[2] = [boxes[box][2], boxes[box][1]]
        points_coords[3] = [boxes[box][2], boxes[box][3]]
        points_coords[4] = [boxes[box][0], boxes[box][3]]

      points_labels[idx] = 1
      points_labels[idx + 1:idx + 5] = 0
      idx += 5
  return points_coords, points_labels
