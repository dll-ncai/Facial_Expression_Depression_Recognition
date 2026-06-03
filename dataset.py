import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AVEC2014Dataset(Dataset):
    """
    AVEC 2014 depression estimation dataset.

    Expects the following directory layout::

        root_dir/
            Training/
                <video_id>/
                    frame_0001.jpg
                    ...
            Development/
                <video_id>/
                    ...
            Testing/
                <video_id>/
                    ...
        labels.csv   (columns: video, split, BDI-II)

    Args:
        root_dir:   Path to the dataset root.
        label_file: Path to the CSV file containing split labels.
        split:      One of ``"Training"``, ``"Development"``, or ``"Testing"``.
        transform:  Torchvision transform applied to every frame.
        max_frames: Maximum number of frames loaded per video.
    """

    def __init__(self, root_dir, label_file, split, transform=None, max_frames=500):
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames

        df = pd.read_csv(label_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
        print(f"[{split}] {len(self.df)} videos loaded.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.root_dir, row["split"], row["video"])
        label = torch.tensor(row["BDI-II"], dtype=torch.float)
        video_id = row["video"]

        if not os.path.exists(video_path):
            return torch.zeros(1, 3, 224, 224), label, video_id

        frame_files = sorted(
            f for f in os.listdir(video_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        frames = []
        for fname in frame_files[: self.max_frames]:
            try:
                img = Image.open(os.path.join(video_path, fname)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception:
                continue

        if not frames:
            frames = [torch.zeros(3, 224, 224)]

        return torch.stack(frames), label, video_id