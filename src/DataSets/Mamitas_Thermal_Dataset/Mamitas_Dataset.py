import os
import zipfile
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import shutil
from typing import Optional, Union, Tuple

class PermuteTensor():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.permute(self.dim)

class ToBoolTensor(transforms.ToTensor):
    def __call__(self, pic):
        tensor = super().__call__(pic)
        tensor = tensor.float()
        return tensor > 0.5

class ToFloatTensor(transforms.ToTensor):
    def __call__(self, pic):
        tensor = super().__call__(pic)
        return tensor.float()

class Mamitas_Create_Dataset(Dataset):
    def __init__(self, 
                 file_images: list, 
                 file_masks: list,
                 merge_image: bool,
                 transform_mask: transforms.Compose,
                 transform_img: transforms.Compose):
        
        self.file_images = file_images
        self.file_masks = file_masks
        self.transform_mask = transform_mask
        self.transform_img = transform_img
        self.merge_image = merge_image

    def __len__(self) -> int:
        return len(self.file_images)

    def __getitem__(self, 
                    idx: int) -> Tuple[torch.Tensor,torch.Tensor,str]:
        root_name_img = self.file_images[idx]
        root_name_mask = self.file_masks[idx]

        img, mask, id_image = Mamitas_Thermal_Feet_Dataset.load_instance(root_name_img,
                                                                         root_name_mask,
                                                                         self.merge_image,
                                                                         self.transform_mask,
                                                                         self.transform_img)

        return img, mask, id_image

class Mamitas_Thermal_Feet_Dataset():
    def __init__(self,credentials_path):
        self.__path_file = os.path.abspath(__file__)
        self.final_path = os.path.join(os.path.dirname(self.__path_file),'data\\')
        self.final_path_zip = os.path.join(os.path.dirname(self.__path_file),'mamitas-thermal-feet.zip')
        self.file_imgs = []
        self.file_masks = []
        self.credentials_path = credentials_path
        
        self.download_by_kaggle()

        for carpet in os.listdir(self.final_path):
            for files in os.listdir(os.path.join(self.final_path, carpet, 'Imágenes')):
                path_file = os.path.join(self.final_path, carpet, 'Imágenes', files)
                self.file_imgs.append(path_file)

                self.file_masks.append(os.path.join(self.final_path, carpet, 'Máscaras - Manuales', path_file.split(os.sep)[-1][:-4] + ".png"))
            #for files in os.listdir(os.path.join(self.final_path, carpet, 'Máscaras - Manuales')):
                #self.file_masks.append(os.path.join(self.final_path, carpet, 'Máscaras - Manuales', files))
        assert len(self.file_imgs) == len(self.file_masks), 'El número de imágenes y sus máscaras no coinciden.'

    def download_by_kaggle(self):
      try:
            credential = self.credentials_path
            if not os.path.isfile(credential):
                raise FileNotFoundError(f"No se encontró el archivo kaggle.json en {credential}")
            dest_folder = os.path.expanduser('~/.kaggle/')
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(credential, dest_folder)
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            dataset_id = 'lucasiturriago/mamitas-thermal-feet'
            api.dataset_download_files(dataset_id,path=os.path.join(os.path.dirname(self.__path_file)+ '\\'))
            with zipfile.ZipFile(self.final_path_zip, 'r') as zip_ref:
                os.makedirs(self.final_path, exist_ok=True)
                zip_ref.extractall(self.final_path)
        
      except FileNotFoundError as e:
          print(f"Error: {e}")
        
      except Exception as e:
          print(f"Error al cargar las credenciales o descargar el dataset: {e}")

    @staticmethod
    def load_instance(root_name_img: str,
                      root_name_mask: str,
                      merge_image: bool,
                      transform_mask: transforms.Compose,
                      transform_img: transforms.Compose) -> Tuple[torch.Tensor,torch.Tensor,str]:
        """
        Load an instance from the dataset

        Args:
          root_name_img (str): Path to image file
          root_name_mask (str): Path to mask file

        Returns:
          img (torch.Tensor): Image tensor
          mask (torch.Tensor): Mask tensor
        """

        img = Image.open(root_name_img).convert('L')
        mask = Image.open(root_name_mask).convert('L')
        if merge_image:
           img = Image.merge('RGB',(img,img,img))
        if transform_mask is not None:
          mask = Mamitas_Thermal_Feet_Dataset.__preprocessing_mask(mask,transform_mask)
        if transform_img is not None:
          img = Mamitas_Thermal_Feet_Dataset.__preprocessing_img(img,transform_img)
        id_image = Mamitas_Thermal_Feet_Dataset.extract_id(root_name_img)
        return img, mask, id_image

    @staticmethod
    def __preprocessing_mask(mask: Image.Image,
                             transforms: transforms.Compose) -> torch.Tensor:
        """
        Transforms a PIL Image

        Args:
            img (PIL Image): PIL Image to transform
        Returns:
            PIL Image: Transformed PIL Image
        """
        #transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            #ToBoolTensor(),
        #])
        mask = transforms(mask)
        return mask

    @staticmethod
    def __preprocessing_img(img: Image.Image,
                            transforms: transforms.Compose) -> torch.Tensor:
      """
      Transforms a PIL Image

      Args:
          img (PIL Image): PIL Image to transform
      Returns:
          PIL Image: Transformed PIL Image
      """
      #transform = transforms.Compose([
          #transforms.Resize((224, 224)),
          #ToFloatTensor(),
      #])
      img = transforms(img)
      return img

    @staticmethod
    def extract_id(path: str) -> str:
      """
      Extract de id for image file

      Args:
        path (str): Path to image file
      Returns:
        str: Id for image file
      """
      path_parts = os.path.normpath(path).split(os.sep)
      case_part = path_parts[-3]
      image_name = os.path.splitext(path_parts[-1])[0]
      result = case_part + image_name
      return result

    def __call__(self):
        return self

    def generate_dataset_with_val(self,
                                  transform_mask: transforms.Compose,
                                  transform_img: transforms.Compose,
                                  merge_image: bool = True,
                                  torch_dataset: bool = False,
                                  batch_size: Optional[int] = 32,
                                  shuffle: Optional[bool] = True,
                                  split_val: float=0.2,
                                  seed:int = 42) -> Union[Tuple[DataLoader,DataLoader],Tuple[Dataset,Dataset]]:
      """
      Generate a dataset with validation split

      Args:
        torch_dataset (bool): If True, returns a torch Dataset object (Optional).
        batch_size (int): Batch size for the dataset (Optional).
        shuffle (bool): Whether to shuffle the dataset (Optional).
        split_val (float): Validation split ratio.
        seed (int): Random seed for shuffling.

      Returns:
        Dataset or DataLoader object.
      """

      train_imgs, val_imgs, train_masks, val_masks = train_test_split(self.file_imgs,
                                                                      self.file_masks,
                                                                      test_size=split_val,
                                                                      random_state=seed)

      print(f"Train_dataset: {len(train_imgs)}")
      print(f"Val_dataset: {len(val_imgs)}")
      train_dataset = Mamitas_Create_Dataset(train_imgs,
                                             train_masks,
                                             merge_image,
                                             transform_mask,
                                             transform_img)

      val_dataset = Mamitas_Create_Dataset(val_imgs,
                                           val_masks,
                                           merge_image,
                                           transform_mask,
                                           transform_img)

      if torch_dataset:
        assert batch_size is not None and shuffle is not None, "batch_size and shuffle must be provided when torch_dataset is True"
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

      return train_dataset, val_dataset

    def generate_dataset(self,
                         transform_mask:transforms.Compose,
                         transform_img:transforms.Compose,
                         merge_image: bool = True,
                         torch_dataset: Optional[bool]= False,
                         batch_size: Optional[int] = 32,
                         shuffle: Optional[bool] = True) -> Union[Dataset, DataLoader]:
      """
      Generate a dataset

      Args:
        torch_dataset (bool): If True, returns a torch Dataset object (Optional).
        batch_size (int): Batch size for the dataset (Optional).
        shuffle (bool): Whether to shuffle the dataset (Optional).

      Returns:
        Dataset or DataLoader object.
      """

      dataset = Mamitas_Create_Dataset(self.file_imgs,
                                       self.file_masks,
                                       merge_image,
                                       transform_mask,
                                       transform_img)

      if torch_dataset:
        assert batch_size is not None and shuffle is not None, "batch_size and shuffle must be provided when torch_dataset is True"
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

      return dataset
    

if __name__ == "__main__":
  Mamitas_Thermal_Feet_Dataset()