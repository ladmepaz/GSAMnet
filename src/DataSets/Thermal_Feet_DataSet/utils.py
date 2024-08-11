import os 
import gdown
import zipfile

def unzip(file_path,destination_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

def download_from_drive(id,destination_path):
    url = f"https://drive.google.com/uc?id={id}&confirm=t"
    if os.path.exists(destination_path):
        return None
    gdown.download(url, destination_path, quiet=False)


def listify(value):
    if not isinstance(value, list):
        value = [value]
    return value 
