"""
JEPA training script.

Usage:
    python -m jepa.train
    python -m jepa.train --epochs 50 --batch_size 64 --lr 0.0002
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jepa.model import JEPA
from data.text_dataset import load_huggingface_dataset, create_jepa_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train JEPA on text data')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='Dataset name from HuggingFace')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1',
                        help='Dataset configuration')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to load (None = all)')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of transformer layers in encoders (12 = BERT-Base)')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--predictor_layers', type=int, default=2,
                        help='Number of predictor layers (2-3 recommended, much smaller than encoder)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--ema_momentum', type=float, default=0.996,
                        help='EMA momentum for target encoder')
    
    # Data arguments
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--total_mask_ratio', type=float, default=0.20,
                        help='Total fraction of tokens to mask')
    parser.add_argument('--num_spans', type=int, default=3,
                        help='Number of spans to mask')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=50,
                        help='Log progress every N batches')
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_arg)


def save_checkpoint(model, optimizer, epoch, loss, args, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)
    }
    
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)
    print(f"Saved checkpoint to {latest_path}")
    
    epoch_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, epoch_path)
    print(f"Saved epoch checkpoint to {epoch_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def train_epoch(model, dataloader, optimizer, device, log_every=50):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    loss_components_sum = {
        'repr_loss': 0.0,
        'var_loss': 0.0,
        'cov_loss': 0.0
    }
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        context_ids = batch['context_ids'].to(device)
        context_mask = batch['context_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_mask = batch['target_mask'].to(device)
        num_targets = batch['num_targets']
        
        outputs = model(
            context_ids=context_ids,
            context_mask=context_mask,
            target_ids=target_ids,
            target_mask=target_mask,
            num_targets=num_targets
        )
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        model.update_target_encoder()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        if hasattr(model, '_last_loss_components'):
            for key in loss_components_sum:
                if key in model._last_loss_components:
                    loss_components_sum[key] += model._last_loss_components[key]
        
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'batch_loss': f'{loss.item():.4f}'
        })
        
        if (batch_idx + 1) % log_every == 0:
            with torch.no_grad():
                predicted = outputs['predicted']
                target = outputs['target']
                
                pred_mean = predicted.mean().item()
                pred_std = predicted.std().item()
                target_mean = target.mean().item()
                target_std = target.std().item()
                
                pred_norm = torch.nn.functional.normalize(predicted, p=2, dim=-1)
                target_norm = torch.nn.functional.normalize(target, p=2, dim=-1)
                cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean().item()
            
            avg_components = {k: v / (batch_idx + 1) for k, v in loss_components_sum.items()}
            
            print(f"\n  Batch [{batch_idx + 1}/{num_batches}]")
            print(f"    Loss: {avg_loss:.4f} (repr: {avg_components.get('repr_loss', 0):.4f}, "
                  f"var: {avg_components.get('var_loss', 0):.4f}, "
                  f"cov: {avg_components.get('cov_loss', 0):.4f})")
            print(f"    Embeddings - Pred(μ={pred_mean:.3f}, σ={pred_std:.3f}), "
                  f"Target(μ={target_mean:.3f}, σ={target_std:.3f})")
            print(f"    Cosine Similarity: {cosine_sim:.4f}")
    
    return total_loss / num_batches


def validate(model, dataloader, device):
    """Validate the model with diagnostics."""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    all_cosine_sims = []
    all_pred_stds = []
    all_target_stds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            context_ids = batch['context_ids'].to(device)
            context_mask = batch['context_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_mask = batch['target_mask'].to(device)
            num_targets = batch['num_targets']
            
            outputs = model(
                context_ids=context_ids,
                context_mask=context_mask,
                target_ids=target_ids,
                target_mask=target_mask,
                num_targets=num_targets
            )
            loss = outputs['loss']
            total_loss += loss.item()
            
            predicted = outputs['predicted']
            target = outputs['target']
            
            pred_norm = torch.nn.functional.normalize(predicted, p=2, dim=-1)
            target_norm = torch.nn.functional.normalize(target, p=2, dim=-1)
            cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean().item()
            all_cosine_sims.append(cosine_sim)
            
            all_pred_stds.append(predicted.std().item())
            all_target_stds.append(target.std().item())
    
    avg_cosine_sim = sum(all_cosine_sims) / len(all_cosine_sims)
    avg_pred_std = sum(all_pred_stds) / len(all_pred_stds)
    avg_target_std = sum(all_target_stds) / len(all_target_stds)
    
    diagnostics = {
        'cosine_sim': avg_cosine_sim,
        'pred_std': avg_pred_std,
        'target_std': avg_target_std
    }
    
    return total_loss / num_batches, diagnostics


def main():
    """Main training function."""
    args = parse_args()
    
    print("="*60)
    print("JEPA Training")
    print("="*60)
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    device = get_device(args.device)
    print(f"Using device: {device}\n")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"✓ Loaded tokenizer (vocab size: {tokenizer.vocab_size})\n")
    
    print(f"Loading dataset: {args.dataset} ({args.dataset_config})...")
    texts = load_huggingface_dataset(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split='train',
        max_samples=args.max_samples
    )
    print(f"✓ Loaded {len(texts)} text samples\n")
    
    print("Creating dataloaders...")
    train_loader, val_loader = create_jepa_dataloaders(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        total_mask_ratio=args.total_mask_ratio,
        num_spans=args.num_spans,
        num_workers=args.num_workers
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}\n")
    
    print("Initializing JEPA model...")
    
    model = JEPA(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.num_layers,
        encoder_heads=args.num_heads,
        predictor_layers=args.predictor_layers,
        ema_momentum=args.ema_momentum
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01
    )
    
    print("="*60)
    print("Starting Training")
    print("="*60)
    print()
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            log_every=args.log_every
        )
        
        val_loss, val_diagnostics = validate(
            model=model,
            dataloader=val_loader,
            device=device
        )
        
        if epoch > args.warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print("  Validation Diagnostics:")
        print(f"    Cosine Similarity: {val_diagnostics['cosine_sim']:.4f}")
        print(f"    Pred Std Dev: {val_diagnostics['pred_std']:.4f}")
        print(f"    Target Std Dev: {val_diagnostics['target_std']:.4f}")
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'val_cosine_sim': val_diagnostics['cosine_sim'],
            'val_pred_std': val_diagnostics['pred_std'],
            'val_target_std': val_diagnostics['target_std']
        })
        
        if epoch % args.save_every == 0 or val_loss < best_val_loss:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                args=args,
                checkpoint_dir=args.checkpoint_dir
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("  ✓ New best validation loss!")
        
        history_path = Path(args.checkpoint_dir) / 'training_history.json'
        history_path.parent.mkdir(exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()

