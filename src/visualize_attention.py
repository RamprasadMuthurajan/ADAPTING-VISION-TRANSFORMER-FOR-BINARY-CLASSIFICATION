"""
Vision Transformer Attention Map Visualization
Usage:
    python visualize_attention.py --model_path output/vitb16_ants_bees_step1000_acc0.9739.bin --image_path my_dataset/val/ants/123.jpg --model_type ViT-B_16
"""

import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS


def get_attention_maps(model, image, device):
    """Extract attention maps from ViT model"""
    model.eval()
    
    # Enable attention weight collection
    attention_weights = []
    
    def hook_fn(module, input, output):
        # output[1] contains attention weights
        if len(output) > 1 and output[1] is not None:
            attention_weights.append(output[1].detach().cpu())
    
    # Register hooks on attention layers
    hooks = []
    for block in model.transformer.encoder.layer:
        hook = block.attn.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        logits, attn = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_weights, logits


def visualize_attention_map(image, attention_weights, layer_idx=11, head_idx=0, patch_size=16):
    """
    Visualize attention map for specific layer and head
    
    Args:
        image: Original PIL Image
        attention_weights: List of attention weight tensors from all layers
        layer_idx: Which transformer layer (0-11 for ViT-B/16)
        head_idx: Which attention head (0-11 for ViT-B/16)
        patch_size: Patch size (16 for ViT-B/16)
    """
    # Get attention weights for specified layer
    attn = attention_weights[layer_idx]  # Shape: [batch, heads, tokens, tokens]
    attn = attn[0]  # Remove batch dimension
    
    # Get attention from CLS token (first token) to all patches
    cls_attn = attn[head_idx, 0, 1:]  # Skip CLS token itself
    
    # Reshape to spatial dimensions
    num_patches = int(np.sqrt(cls_attn.shape[0]))
    attn_map = cls_attn.reshape(num_patches, num_patches).numpy()
    
    # Normalize attention map
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # Resize attention map to image size
    img_size = image.size
    attn_map_resized = cv2.resize(attn_map, img_size, interpolation=cv2.INTER_CUBIC)
    
    return attn_map_resized


def attention_rollout(attention_weights, start_layer=0, discard_ratio=0.1):
    """
    Attention rollout method to aggregate attention across layers
    Reference: https://arxiv.org/abs/2005.00928
    """
    result = torch.eye(attention_weights[0].shape[-1])
    
    for attn in attention_weights[start_layer:]:
        # Average attention across heads
        attn_heads_fused = attn.mean(dim=1)  # [batch, tokens, tokens]
        attn_heads_fused = attn_heads_fused[0]  # Remove batch dim
        
        # Drop lowest attentions
        flat = attn_heads_fused.view(-1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), largest=False)
        flat[indices] = 0
        
        # Normalize
        eye = torch.eye(attn_heads_fused.size(-1))
        a = (attn_heads_fused + eye) / 2
        a = a / a.sum(dim=-1, keepdim=True)
        
        result = torch.matmul(a, result)
    
    # Get attention from CLS token
    mask = result[0, 1:]  # CLS to patches
    mask = mask / mask.max()
    
    return mask.numpy()


def plot_attention_grid(image, attention_weights, save_path, num_layers=4):
    """Plot attention maps from multiple layers"""
    fig, axes = plt.subplots(2, num_layers, figsize=(num_layers*4, 8))
    
    # Plot original image
    for i in range(num_layers):
        axes[0, i].imshow(image)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original Image')
    
    # Plot attention maps from different layers
    layer_indices = np.linspace(0, len(attention_weights)-1, num_layers, dtype=int)
    
    for idx, layer_idx in enumerate(layer_indices):
        attn_map = visualize_attention_map(image, attention_weights, layer_idx=layer_idx, head_idx=0)
        
        axes[1, idx].imshow(image)
        axes[1, idx].imshow(attn_map, cmap='jet', alpha=0.6)
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f'Layer {layer_idx}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved attention grid to {save_path}")
    plt.close()


def plot_attention_heads(image, attention_weights, layer_idx, save_path, num_heads=12):
    """Plot attention maps from all heads in a specific layer"""
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        attn_map = visualize_attention_map(image, attention_weights, layer_idx=layer_idx, head_idx=head_idx)
        
        axes[head_idx].imshow(image)
        axes[head_idx].imshow(attn_map, cmap='jet', alpha=0.6)
        axes[head_idx].axis('off')
        axes[head_idx].set_title(f'Head {head_idx}')
    
    plt.suptitle(f'Attention Heads - Layer {layer_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved attention heads to {save_path}")
    plt.close()


def plot_attention_rollout(image, attention_weights, save_path):
    """Plot attention rollout visualization"""
    mask = attention_rollout(attention_weights, discard_ratio=0.1)
    
    # Reshape to spatial dimensions
    num_patches = int(np.sqrt(mask.shape[0]))
    mask = mask.reshape(num_patches, num_patches)
    
    # Resize to image size
    img_size = image.size
    mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_CUBIC)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Original Image', fontsize=12)
    
    # Attention heatmap
    axes[1].imshow(mask_resized, cmap='jet')
    axes[1].axis('off')
    axes[1].set_title('Attention Rollout Heatmap', fontsize=12)
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(mask_resized, cmap='jet', alpha=0.6)
    axes[2].axis('off')
    axes[2].set_title('Overlay', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved attention rollout to {save_path}")
    plt.close()


def load_model(model_path, model_type, device):
    """Load trained model"""
    config = CONFIGS[model_type]
    model = VisionTransformer(config, img_size=224, num_classes=2, zero_head=False, vis=True)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from {model_path}")
    return model


def preprocess_image(image_path):
    """Preprocess image for ViT"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    return image, image_tensor


def main():
    parser = argparse.ArgumentParser(description="Visualize ViT Attention Maps")
    
    parser.add_argument("--model_path", required=True, help="Path to trained model .bin file")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--model_type", default="ViT-B_16", 
                       choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "R50-ViT-B_16"],
                       help="Model architecture")
    parser.add_argument("--output_dir", default="attention_visualizations", help="Output directory")
    parser.add_argument("--layer", type=int, default=11, help="Layer to visualize (0-11 for ViT-B)")
    
    args = parser.parse_args()
    
    # Setup
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("🔍 ATTENTION MAP VISUALIZATION")
    print("="*60)
    print(f"Model:      {args.model_path}")
    print(f"Image:      {args.image_path}")
    print(f"Model type: {args.model_type}")
    print(f"Device:     {device}")
    print("="*60 + "\n")
    
    # Load model
    model = load_model(args.model_path, args.model_type, device)
    
    # Load and preprocess image
    image, image_tensor = preprocess_image(args.image_path)
    print(f"📷 Loaded image: {image.size}")
    
    # Get attention maps
    print("🔄 Computing attention maps...")
    attention_weights, logits = get_attention_maps(model, image_tensor, device)
    
    # Get prediction
    pred_class = torch.argmax(logits, dim=1).item()
    pred_prob = torch.softmax(logits, dim=1)[0, pred_class].item()
    class_names = ['Ants', 'Bees']
    print(f"🎯 Prediction: {class_names[pred_class]} ({pred_prob*100:.2f}%)\n")
    
    # Generate visualizations
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # 1. Attention grid across layers
    print("📊 Generating attention grid across layers...")
    plot_attention_grid(
        image, 
        attention_weights, 
        os.path.join(args.output_dir, f"{base_name}_attention_grid.png")
    )
    
    # 2. All attention heads in specific layer
    print("📊 Generating attention heads visualization...")
    plot_attention_heads(
        image, 
        attention_weights, 
        layer_idx=args.layer,
        save_path=os.path.join(args.output_dir, f"{base_name}_attention_heads_layer{args.layer}.png")
    )
    
    # 3. Attention rollout
    print("📊 Generating attention rollout...")
    plot_attention_rollout(
        image,
        attention_weights,
        os.path.join(args.output_dir, f"{base_name}_attention_rollout.png")
    )
    
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()