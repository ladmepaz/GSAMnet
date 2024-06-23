from groundingdino.SamLangDino import SamLangDino
from DataSets.Mamitas_Thermal_Dataset.Mamitas_Dataset import Mamitas_Thermal_Feet_Dataset, ToBoolTensor, PermuteTensor
from torchvision.transforms import transforms
import torch
transform_mask = transforms.Compose([
    ToBoolTensor(),
    PermuteTensor((1,2,0))
])

transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.uint8)
])

train_dataset, val_dataset = Mamitas_Thermal_Feet_Dataset().generate_dataset_with_val(transform_mask=transform_mask,
                                                                                      transform_img=transform_img,
                                                                                      split_val=0.1,
                                                                                      shuffle=True,
                                                                                      seed=42)
model = SamLangDino()


