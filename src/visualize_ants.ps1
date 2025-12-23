# ViT-B_16, 224x224
python visualize_attention.py `
  --model_path output/vitb16_224_ants_bees_step100_acc0.9869.bin `
  --image_path "my_dataset/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg" `
  --model_type ViT-B_16

# ViT-B_16, 384x384
python visualize_attention.py `
  --model_path output/vitb16_384_ants_bees_step200_acc0.9869.bin `
  --image_path "my_dataset/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg" `
  --model_type ViT-B_16

# ViT-B_32, 384x384
python visualize_attention.py `
  --model_path output/vitb32_384_ants_bees_step100_acc0.9739.bin `
  --image_path "my_dataset/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg" `
  --model_type ViT-B_32

# ViT-L_16, 224x224
python visualize_attention.py `
  --model_path output/vitl16_224_ants_bees_step100_acc0.9869.bin `
  --image_path "my_dataset/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg" `
  --model_type ViT-L_16

# ViT-L_16, 384x384
python visualize_attention.py `
  --model_path output/vitl16_384_ants_bees_step200_acc0.9869.bin `
  --image_path "my_dataset/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg" `
  --model_type ViT-L_16

# ViT-L_32, 384x384
python visualize_attention.py `
  --model_path output/vitl32_384_ants_bees_step100_acc0.9739.bin `
  --image_path "my_dataset/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg" `
  --model_type ViT-L_32

# R50-ViT-B_16, 384x384
python visualize_attention.py `
  --model_path output/r50vitb16_384_ants_bees_step200_acc0.9608.bin `
  --image_path "my_dataset/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg" `
  --model_type R50-ViT-B_16