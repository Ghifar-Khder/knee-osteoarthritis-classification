import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import time
from tqdm import tqdm
import json
from datetime import datetime

class Config:
    BASE_PATH = r"Data\KNEE-images\train"
    OUTPUT_PATH = r"D:\AAA-projects-to-do\KNEE\GAN-original"
    
    # Sub-directories for organized output
    ORIGINAL_RESIZED_PATH = os.path.join(OUTPUT_PATH, "original_128")
    GENERATED_PATH = os.path.join(OUTPUT_PATH, "generated")
    MODELS_PATH = os.path.join(OUTPUT_PATH, "models")
    
    IMG_SIZE = 128
    NUM_CLASSES = 5
    LATENT_DIM = 100
    BATCH_SIZE = 16
    G_LR = 0.0002
    D_LR = 0.0002
    EPOCHS = 150
    CLASS_WEIGHTS = [1.0, 1.0, 2.0, 2.1, 2.7]
    TARGET_COUNTS = {0: 2300, 1: 2300, 2: 2300, 3: 2300, 4: 2300}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

def setup_directories():
    """Create organized output directories"""
    directories = [
        config.OUTPUT_PATH,
        config.ORIGINAL_RESIZED_PATH,
        config.GENERATED_PATH,
        config.MODELS_PATH
    ]
    
    # Create class subdirectories
    for class_idx in range(config.NUM_CLASSES):
        directories.append(os.path.join(config.ORIGINAL_RESIZED_PATH, str(class_idx)))
        directories.append(os.path.join(config.GENERATED_PATH, str(class_idx)))
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def resize_and_save_original_images():
    """Resize original images to 128x128 and save them"""
    print("Resizing and saving original images...")
    
    for class_idx in tqdm(range(config.NUM_CLASSES), desc="Classes"):
        class_dir = os.path.join(config.BASE_PATH, str(class_idx))
        save_dir = os.path.join(config.ORIGINAL_RESIZED_PATH, str(class_idx))
        
        if not os.path.exists(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for img_file in tqdm(image_files, desc=f"KL{class_idx}", leave=False):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load and resize image
                img = Image.open(img_path).convert('L')
                img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
                
                # Save resized image
                save_path = os.path.join(save_dir, f"resized_{img_file}")
                img.save(save_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

class KneeXRayDataset(Dataset):
    def __init__(self, root_dir, use_resized=True):
        self.root_dir = config.ORIGINAL_RESIZED_PATH if use_resized else root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.images = []
        self.labels = []
        
        for label in range(config.NUM_CLASSES):
            class_dir = os.path.join(self.root_dir, str(label))
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                for img_file in image_files:
                    self.images.append(os.path.join(class_dir, img_file))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        label_onehot = torch.zeros(config.NUM_CLASSES)
        label_onehot[label] = 1
        weight = config.CLASS_WEIGHTS[label]
        return img, label, label_onehot, weight

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=5):
        super().__init__()
        self.label_embed = nn.Linear(num_classes, 32)
        self.init_size = 8
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 32, 512 * self.init_size * self.init_size),
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
        z_combined = torch.cat([z, label_emb], dim=1)
        out = self.fc(z_combined)
        out = out.view(-1, 512, self.init_size, self.init_size)
        out = self.up1(out)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)
        image = self.tanh(self.image_head(out))
        return image

class Discriminator(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.label_embed = nn.Linear(num_classes, config.IMG_SIZE * config.IMG_SIZE)
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, img, labels=None):
        batch_size = img.size(0)
        if labels is not None:
            label_emb = self.label_embed(labels)
            label_emb = label_emb.view(batch_size, 1, config.IMG_SIZE, config.IMG_SIZE)
            img = torch.cat([img, label_emb], dim=1)
        else:
            zeros = torch.zeros(batch_size, 1, config.IMG_SIZE, config.IMG_SIZE).to(img.device)
            img = torch.cat([img, zeros], dim=1)
        
        features = self.model[:-2](img)
        real_fake = self.model[-2:](features)
        class_pred = self.classifier(features)
        return real_fake, class_pred

class GANTrainer:
    def __init__(self):
        self.device = config.DEVICE
        self.generator = ConditionalGenerator(
            latent_dim=config.LATENT_DIM,
            num_classes=config.NUM_CLASSES
        ).to(self.device)
        
        self.discriminator = Discriminator(
            num_classes=config.NUM_CLASSES
        ).to(self.device)
        
        self.g_optim = optim.Adam(self.generator.parameters(), lr=config.G_LR, betas=(0.5, 0.999))
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=config.D_LR, betas=(0.5, 0.999))
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.CrossEntropyLoss()
    
    def save_models(self, epoch=None):
        """Save generator and discriminator models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if epoch is not None:
            gen_path = os.path.join(config.MODELS_PATH, f"generator_epoch_{epoch}_{timestamp}.pth")
            disc_path = os.path.join(config.MODELS_PATH, f"discriminator_epoch_{epoch}_{timestamp}.pth")
        else:
            gen_path = os.path.join(config.MODELS_PATH, f"generator_final_{timestamp}.pth")
            disc_path = os.path.join(config.MODELS_PATH, f"discriminator_final_{timestamp}.pth")
        
        torch.save({
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.g_optim.state_dict(),
            'epoch': epoch if epoch is not None else config.EPOCHS
        }, gen_path)
        
        torch.save({
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.d_optim.state_dict(),
            'epoch': epoch if epoch is not None else config.EPOCHS
        }, disc_path)
        
        print(f"Models saved: {gen_path}")
    
    def load_models(self, gen_path, disc_path):
        """Load generator and discriminator models"""
        gen_checkpoint = torch.load(gen_path, map_location=self.device)
        disc_checkpoint = torch.load(disc_path, map_location=self.device)
        
        self.generator.load_state_dict(gen_checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(disc_checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in gen_checkpoint:
            self.g_optim.load_state_dict(gen_checkpoint['optimizer_state_dict'])
        if 'optimizer_state_dict' in disc_checkpoint:
            self.d_optim.load_state_dict(disc_checkpoint['optimizer_state_dict'])
        
        print(f"Models loaded from {gen_path}")
    
    def train_step(self, real_images, real_labels, real_labels_onehot):
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        real_labels = real_labels.to(self.device)
        real_labels_onehot = real_labels_onehot.to(self.device)
        
        z = torch.randn(batch_size, config.LATENT_DIM).to(self.device)
        fake_images = self.generator(z, real_labels_onehot)
        
        # Train Discriminator
        self.d_optim.zero_grad()
        real_real_fake, real_class_pred = self.discriminator(real_images, real_labels_onehot)
        fake_real_fake, fake_class_pred = self.discriminator(fake_images.detach(), real_labels_onehot)
        
        real_target = torch.ones_like(real_real_fake).to(self.device)
        fake_target = torch.zeros_like(fake_real_fake).to(self.device)
        
        d_real_loss = self.adversarial_loss(real_real_fake, real_target)
        d_fake_loss = self.adversarial_loss(fake_real_fake, fake_target)
        d_adv_loss = (d_real_loss + d_fake_loss) / 2
        d_class_loss = self.class_loss(real_class_pred, real_labels)
        d_loss = d_adv_loss + 0.5 * d_class_loss
        d_loss.backward()
        self.d_optim.step()
        
        # Train Generator
        self.g_optim.zero_grad()
        fake_images = self.generator(z, real_labels_onehot)
        fake_real_fake, fake_class_pred = self.discriminator(fake_images, real_labels_onehot)
        
        g_adv_loss = self.adversarial_loss(fake_real_fake, real_target)
        g_class_loss = self.class_loss(fake_class_pred, real_labels)
        g_loss = g_adv_loss + 0.7 * g_class_loss
        g_loss.backward()
        self.g_optim.step()
        
        with torch.no_grad():
            _, pred_classes = torch.max(fake_class_pred, 1)
            class_acc = (pred_classes == real_labels).float().mean()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'class_acc': class_acc.item()
        }
    
    def train(self, dataloader, epochs=config.EPOCHS):
        print(f"Training for {epochs} epochs")
        
        history = {
            'epoch': [],
            'd_loss': [],
            'g_loss': [],
            'accuracy': []
        }
        
        epoch_pbar = tqdm(range(epochs), desc="Training Progress")
        
        for epoch in epoch_pbar:
            epoch_d_loss = 0
            epoch_g_loss = 0
            epoch_acc = 0
            
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch in batch_pbar:
                images, labels, labels_onehot, _ = batch
                metrics = self.train_step(images, labels, labels_onehot)
                
                epoch_d_loss += metrics['d_loss']
                epoch_g_loss += metrics['g_loss']
                epoch_acc += metrics['class_acc']
                
                batch_pbar.set_postfix({
                    'D': f"{metrics['d_loss']:.3f}",
                    'G': f"{metrics['g_loss']:.3f}",
                    'Acc': f"{metrics['class_acc']:.3f}"
                })
            
            batch_pbar.close()
            
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_acc = epoch_acc / len(dataloader)
            
            history['epoch'].append(epoch + 1)
            history['d_loss'].append(avg_d_loss)
            history['g_loss'].append(avg_g_loss)
            history['accuracy'].append(avg_acc)
            
            epoch_pbar.set_postfix({
                'D_loss': f"{avg_d_loss:.4f}",
                'G_loss': f"{avg_g_loss:.4f}",
                'Acc': f"{avg_acc:.4f}"
            })
            
            # Save models and samples at checkpoints
            if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                self.save_models(epoch + 1)
                self.save_samples(epoch + 1)
        
        epoch_pbar.close()
        
        # Save final models
        self.save_models()
        
        return history
    
    def save_samples(self, epoch, num_per_class=5):
        """Save generated samples to generated folder"""
        self.generator.eval()
        
        with torch.no_grad():
            for class_idx in range(config.NUM_CLASSES):
                labels = torch.zeros(num_per_class, config.NUM_CLASSES).to(self.device)
                labels[:, class_idx] = 1
                z = torch.randn(num_per_class, config.LATENT_DIM).to(self.device)
                images = self.generator(z, labels)
                
                save_dir = os.path.join(config.GENERATED_PATH, str(class_idx))
                
                for i in range(num_per_class):
                    img_np = images[i, 0].cpu().numpy()
                    img_np = ((img_np + 1) * 127.5).astype(np.uint8)
                    img_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png")
                    Image.fromarray(img_np).save(img_path)
        
        self.generator.train()
    
    def generate_balancing_images(self, original_counts, target_count=500):
        """Generate images to balance the dataset"""
        self.generator.eval()
        
        total_generated = 0
        
        with torch.no_grad():
            for class_idx in tqdm(range(config.NUM_CLASSES), desc="Generating balancing images"):
                needed = max(0, target_count - original_counts.get(class_idx, 0))
                
                if needed > 0:
                    batch_size = min(32, needed)
                    num_batches = (needed + batch_size - 1) // batch_size
                    
                    for batch_idx in range(num_batches):
                        current_batch = min(batch_size, needed - batch_idx * batch_size)
                        labels = torch.zeros(current_batch, config.NUM_CLASSES).to(self.device)
                        labels[:, class_idx] = 1
                        z = torch.randn(current_batch, config.LATENT_DIM).to(self.device)
                        images = self.generator(z, labels)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        for i in range(images.size(0)):
                            img_np = images[i, 0].cpu().numpy()
                            img_np = ((img_np + 1) * 127.5).astype(np.uint8)
                            save_path = os.path.join(
                                config.GENERATED_PATH,
                                str(class_idx),
                                f"balanced_{timestamp}_{batch_idx}_{i}.png"
                            )
                            Image.fromarray(img_np).save(save_path)
                        
                        total_generated += current_batch
        
        self.generator.train()
        return total_generated

def count_images(path):
    """Count images in directory organized by classes"""
    counts = {}
    if not os.path.exists(path):
        return counts
    
    for class_idx in range(config.NUM_CLASSES):
        class_dir = os.path.join(path, str(class_idx))
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            counts[class_idx] = len(image_files)
        else:
            counts[class_idx] = 0
    
    return counts

def save_training_summary(history, original_counts, generated_counts, total_generated, training_time):
    """Save training summary to JSON"""
    summary = {
        "original_counts": original_counts,
        "generated_counts": generated_counts,
        "target_counts": config.TARGET_COUNTS,
        "total_generated": total_generated,
        "training_epochs": config.EPOCHS,
        "training_time": training_time,
        "final_d_loss": history['d_loss'][-1] if history['d_loss'] else 0,
        "final_g_loss": history['g_loss'][-1] if history['g_loss'] else 0,
        "final_accuracy": history['accuracy'][-1] if history['accuracy'] else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_path = os.path.join(config.OUTPUT_PATH, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for class_idx in range(config.NUM_CLASSES):
        print(f"KL{class_idx}: Original={original_counts.get(class_idx, 0)} | "
              f"Generated={generated_counts.get(class_idx, 0)} | "
              f"Total={original_counts.get(class_idx, 0) + generated_counts.get(class_idx, 0)}")
    print(f"\nTotal generated: {total_generated}")
    print(f"Training time: {training_time:.1f} seconds")
    print(f"Final D_loss: {history['d_loss'][-1]:.4f}")
    print(f"Final G_loss: {history['g_loss'][-1]:.4f}")
    print(f"Final Accuracy: {history['accuracy'][-1]:.4f}")
    print("="*50)

def run_training():
    start_time = time.time()
    
    # Setup directories
    print("Setting up directories...")
    setup_directories()
    
    # Resize and save original images
    resize_and_save_original_images()
    
    # Count original images
    original_counts = count_images(config.ORIGINAL_RESIZED_PATH)
    print("\nOriginal images:")
    for i in range(config.NUM_CLASSES):
        print(f"  KL{i}: {original_counts.get(i, 0)}")
    
    # Create dataset from resized images
    dataset = KneeXRayDataset(config.ORIGINAL_RESIZED_PATH, use_resized=True)
    weights = [config.CLASS_WEIGHTS[label] for label in dataset.labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )
    
    # Train GAN
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    trainer = GANTrainer()
    history = trainer.train(dataloader, epochs=config.EPOCHS)
    
    # Generate balancing images
    print("\nGenerating balancing images...")
    total_generated = trainer.generate_balancing_images(original_counts)
    
    # Count generated images
    generated_counts = count_images(config.GENERATED_PATH)
    
    # Save summary
    training_time = time.time() - start_time
    save_training_summary(history, original_counts, generated_counts, total_generated, training_time)
    
    print(f"\nTraining completed!")
    print(f"Output structure:")
    print(f"  - Original resized images: {config.ORIGINAL_RESIZED_PATH}")
    print(f"  - Generated images: {config.GENERATED_PATH}")
    print(f"  - Saved models: {config.MODELS_PATH}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Knee X-ray GAN Training")
    print("="*50)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Target per class: {config.TARGET_COUNTS[0]}")
    print("="*50)
    
    run_training()