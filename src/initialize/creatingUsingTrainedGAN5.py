import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

# ========================
# CONFIG
# ========================
NUM_CLASSES = 5
LATENT_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENERATOR_PATH = r"GAN-originalDataset\models\generator_final.pth"
OUTPUT_BASE = r"new-images-from-GAN"

IMAGES_TO_GENERATE = {
    0: 14,
    1: 1254,
    2: 784,
    3: 1543,
    4: 2127
}

# ========================
# MODEL DEFINITIONS
# ========================
class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class ConditionalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Linear(NUM_CLASSES, 32)
        self.init_size = 8

        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + 32, 512 * self.init_size * self.init_size),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            AttentionGate(256)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            AttentionGate(128)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            AttentionGate(64)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.image_head = nn.Conv2d(32, 1, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        label_emb = self.label_embed(labels)
        x = torch.cat([z, label_emb], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, self.init_size, self.init_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.tanh(self.image_head(x))

# ========================
# MAIN
# ========================
def main():
    print("=" * 60)
    print("SAFE GAN IMAGE GENERATION (NO OVERWRITES)")
    print("=" * 60)

    for c in range(NUM_CLASSES):
        os.makedirs(os.path.join(OUTPUT_BASE, str(c)), exist_ok=True)

    generator = ConditionalGenerator().to(DEVICE)
    checkpoint = torch.load(GENERATOR_PATH, map_location=DEVICE)
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.eval()

    print("✅ Generator loaded\n")

    with torch.no_grad():
        for class_idx, total_images in IMAGES_TO_GENERATE.items():
            print(f"Class {class_idx} → {total_images} images")

            save_dir = os.path.join(OUTPUT_BASE, str(class_idx))
            img_counter = 0

            batch_size = 32
            num_batches = (total_images + batch_size - 1) // batch_size

            for _ in tqdm(range(num_batches), desc=f"KL{class_idx}"):
                current_bs = min(batch_size, total_images - img_counter)

                z = torch.randn(current_bs, LATENT_DIM, device=DEVICE)
                labels = torch.zeros(current_bs, NUM_CLASSES, device=DEVICE)
                labels[:, class_idx] = 1

                images = generator(z, labels)

                for i in range(current_bs):
                    img = images[i, 0].cpu().numpy()
                    img = ((img + 1) * 127.5).astype(np.uint8)

                    save_path = os.path.join(
                        save_dir,
                        f"gan_{img_counter:05d}.png"
                    )

                    Image.fromarray(img).save(save_path)
                    img_counter += 1

            print(f"✅ Saved exactly {img_counter} images\n")

    print("🎉 GENERATION COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
