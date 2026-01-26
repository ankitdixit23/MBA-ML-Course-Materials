# Last run in environment mbaml with python 3.11.13 on 9/27/25
import os
import tarfile
import urllib.request
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# Get the directory this script is in 
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct data paths relative to the script
data_dir = script_dir  
xray_root = os.path.abspath(os.path.join(data_dir, ".."))  # .../xray/
csv_path = os.path.join(data_dir, "Data_Entry_2017_v2020.csv")
output_path = os.path.join(data_dir, "xray_data.pt")

# Make sure output directory exists
os.makedirs(data_dir, exist_ok=True)

# Load metadata and create label dictionary
metadata = pd.read_csv(csv_path)
metadata = metadata[metadata["Finding Labels"].str.contains("No Finding|Infiltration")]
metadata.loc[
    metadata["Finding Labels"].str.contains("Infiltration"), "Finding Labels"
] = "Infiltration"
label_dict = dict(zip(metadata["Image Index"], metadata["Finding Labels"]))

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Initialize storage
infiltration_imgs, no_finding_imgs = [], []
infiltration_labels, no_finding_labels = [], []

# Download and process
links = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]

for idx, link in enumerate(links):
    if len(infiltration_imgs) >= 3000 and len(no_finding_imgs) >= 3000:
        break

    fn = f"images_{idx + 1:02d}.tar.gz"
    local_path = os.path.join(xray_root, fn)

    print(f"Downloading {fn}...")
    urllib.request.urlretrieve(link, local_path)

    print(f"Processing {fn}...")

    with tarfile.open(local_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers()):
            if len(infiltration_imgs) >= 3000 and len(no_finding_imgs) >= 3000:
                break

            if member.isfile() and member.name.endswith(".png"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    filename = os.path.basename(member.name)
                    label = label_dict.get(filename)
                    if label not in {"Infiltration", "No Finding"}:
                        continue

                    if label == "Infiltration" and len(infiltration_imgs) >= 5000:
                        continue
                    if label == "No Finding" and len(no_finding_imgs) >= 5000:
                        continue

                    image = Image.open(BytesIO(f.read())).convert("RGB")
                    tensor = transform(image)

                    if label == "Infiltration":
                        infiltration_imgs.append(tensor)
                        infiltration_labels.append(1)
                    else:
                        no_finding_imgs.append(tensor)
                        no_finding_labels.append(0)
                except Exception as e:
                    print(f"Error processing {member.name}: {e}")

    os.remove(local_path)
    print(f"Deleted {fn} to save space.")

# Combine and save
all_images = torch.stack(infiltration_imgs + no_finding_imgs)
all_labels = torch.tensor(infiltration_labels + no_finding_labels)

# Shuffle
perm = torch.randperm(len(all_labels))
all_images = all_images[perm]
all_labels = all_labels[perm]

torch.save({"images": all_images, "labels": all_labels}, output_path)
print(f"Saved 6,000 samples to: {output_path}")
