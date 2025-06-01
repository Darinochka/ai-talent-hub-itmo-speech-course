import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import torchaudio
from tqdm import tqdm
import os
from models import Generator, Discriminator
from t2spec_converter import TextToSpecConverter
import torch.multiprocessing as mp
import threading

mp.set_start_method('spawn', force=True)

t2s_local = threading.local()

def worker_init_fn(worker_id):
    t2s_local.t2s = TextToSpecConverter()

class VocoderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sample_rate=22050, max_frames=1000, max_samples=22050*5):
        self.dataset = torchaudio.datasets.LJSPEECH(root=root_dir, download=True)
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.max_samples = max_samples
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate, text, _ = self.dataset[idx]
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
        
        t2s = getattr(t2s_local, 't2s', None)
        if t2s is None:
            raise RuntimeError("TextToSpecConverter not initialized in worker")
        mel_spec = t2s.text2spec(text)
        mel_spec = torch.FloatTensor(mel_spec)
        
        T, C = mel_spec.shape
        if T > self.max_frames:
            mel_spec = mel_spec[:self.max_frames, :]
        elif T < self.max_frames:
            pad_amount = self.max_frames - T
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 0, 0, pad_amount))
        
        waveform = waveform.squeeze()
        T = waveform.shape[0]
        if T > self.max_samples:
            waveform = waveform[:self.max_samples]
        elif T < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - T))
        
        return {
            'waveform': waveform,
            'mel_spec': mel_spec,
            'text': text
        }

def custom_collate_fn(batch):
    waveforms = torch.stack([item['waveform'] for item in batch])
    mel_specs = torch.stack([item['mel_spec'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'waveform': waveforms,
        'mel_spec': mel_specs,
        'text': texts
    }

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(
        project="neural-vocoder",
        config={
            "g_learning_rate": args.g_learning_rate,
            "d_learning_rate": args.d_learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lambda_feat": args.lambda_feat,
            "lambda_l1": args.lambda_l1
        }
    )
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    g_optimizer = optim.AdamW(generator.parameters(), lr=args.g_learning_rate, betas=(0.8, 0.99))
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=args.d_learning_rate, betas=(0.8, 0.99))
    
    dataset = VocoderDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=None,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_init_fn
    )
    
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in pbar:
            mel_spec = batch['mel_spec'].transpose(1, 2).to(device)
            waveform = batch['waveform'].to(device)
            
            d_optimizer.zero_grad()
            
            fake_audio = generator(mel_spec)
            fake_audio = fake_audio[:, :, :waveform.size(1)]
            
            real_pred, real_features = discriminator(waveform.unsqueeze(1))
            real_features = [f.detach() for f in real_features]
            
            fake_pred, fake_features = discriminator(fake_audio.detach())
            
            d_loss_real = torch.mean((real_pred - 1) ** 2)
            d_loss_fake = torch.mean(fake_pred ** 2)
            d_loss = d_loss_real + d_loss_fake
            
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            
            fake_pred, fake_features = discriminator(fake_audio)
            
            g_loss_adv = torch.mean((fake_pred - 1) ** 2)
            g_loss_feat = feature_loss(real_features, fake_features)
            g_loss_l1 = nn.L1Loss()(fake_audio.squeeze(), waveform)
            
            g_loss = g_loss_adv + args.lambda_feat * g_loss_feat + args.lambda_l1 * g_loss_l1
            
            g_loss.backward()
            g_optimizer.step()
            
            pbar.set_postfix({
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item()
            })
            
            step = epoch * len(dataloader) + pbar.n
            wandb.log({
                'Loss/Discriminator': d_loss.item(),
                'Loss/Generator': g_loss.item(),
                'Loss/Generator_Adv': g_loss_adv.item(),
                'Loss/Generator_Feat': g_loss_feat.item(),
                'Loss/Generator_L1': g_loss_l1.item(),
                'epoch': epoch,
                'step': step
            })
        
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{epoch+1}.pt')
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_optimizer': d_optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        
        wandb.save(checkpoint_path)
    
    wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to LJSpeech dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--g_learning_rate', type=float, default=2e-4, help='Generator learning rate')
    parser.add_argument('--d_learning_rate', type=float, default=2e-4, help='Discriminator learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval in epochs')
    parser.add_argument('--lambda_feat', type=float, default=2.0, help='Feature matching loss weight')
    parser.add_argument('--lambda_l1', type=float, default=45.0, help='L1 loss weight')
    
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train(args)