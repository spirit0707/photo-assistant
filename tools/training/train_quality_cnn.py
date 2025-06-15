import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

IMG_SIZE = 224  
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = 'core/ml/quality/quality_cnn.pt'
SAMPLES_DIR = 'data/samples/quality_samples'
LABELS_CSV = 'data/labels/quality_labels.csv'

class QualityDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        score = float(row['score'])
        return image, torch.tensor([score], dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    df = pd.read_csv(LABELS_CSV)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_frac = 0.15
    val_size = int(len(df) * val_frac)
    train_df = df[:-val_size]
    val_df = df[-val_size:]
    train_ds = QualityDataset(train_df, SAMPLES_DIR, transform)
    val_ds = QualityDataset(val_df, SAMPLES_DIR, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)  # регрессия
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, scores in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - train'):
            images, scores = images.to(device), scores.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, scores in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - val'):
                images, scores = images.to(device), scores.to(device)
                outputs = model(images)
                loss = criterion(outputs, scores)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'Лучшая модель сохранена в {MODEL_PATH}')

    print('Обучение завершено.')

if __name__ == '__main__':
    main() 