# save_metrics.py
import evaluate
import os

# 定义保存指标脚本的目录
save_directory = "./local_metrics"
os.makedirs(save_directory, exist_ok=True)

print("Downloading 'rouge' metric...")
try:
    rouge = evaluate.load("rouge")
    rouge.save_to_disk(os.path.join(save_directory, "rouge"))
    print("'rouge' metric saved successfully.")
except Exception as e:
    print(f"Failed to download 'rouge': {e}")

print("\nDownloading 'bleu' metric...")
try:
    bleu = evaluate.load("bleu")
    bleu.save_to_disk(os.path.join(save_directory, "bleu"))
    print("'bleu' metric saved successfully.")
except Exception as e:
    print(f"Failed to download 'bleu': {e}")

print(f"\nMetrics have been saved to the '{save_directory}' directory.")
print("Please copy this entire 'local_metrics' folder to your offline server.")
