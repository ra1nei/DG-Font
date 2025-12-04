import os
import torch
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance


# --- Utility functions ---
def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)


def l1_loss(img1, img2):
    return torch.mean(torch.abs(img1 - img2)).item()


def ssim_score(img1, img2):
    img1 = img1.squeeze().permute(1, 2, 0).numpy()
    img2 = img2.squeeze().permute(1, 2, 0).numpy()
    return ssim(img1, img2, channel_axis=2, data_range=1.0)


# --- Main evaluation ---
def evaluate_folder(folder_path, output_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    files = os.listdir(folder_path)
    generated_files = [f for f in files if "|generated_images" in f]

    # Setup models
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    results = []
    gen_imgs = []
    gt_imgs = []

    for gen_file in tqdm(generated_files, desc="Evaluating"):
        base_name = gen_file.replace("|generated_images.png", "")
        gt_file = base_name + "|gt_images.png"

        gen_path = os.path.join(folder_path, gen_file)
        gt_path = os.path.join(folder_path, gt_file)
        if not os.path.exists(gt_path):
            print(f"❌ Missing GT for {gen_file}, skipped.")
            continue

        gen_img = load_image(gen_path).to(device)
        gt_img = load_image(gt_path).to(device)

        # Store for FID later
        gen_imgs.append(gen_img)
        gt_imgs.append(gt_img)

        # --- Per-image metrics ---
        l1_val = l1_loss(gen_img, gt_img)
        ssim_val = ssim_score(gen_img.cpu(), gt_img.cpu())
        lpips_val = lpips_model(gen_img, gt_img).item()

        results.append((base_name, l1_val, ssim_val, lpips_val))

    # --- Compute FID (global) ---
    for img in gen_imgs:
        fid_metric.update((img * 255).byte(), real=False)
    for img in gt_imgs:
        fid_metric.update((img * 255).byte(), real=True)

    print("Gen:", len(gen_imgs), "GT:", len(gt_imgs))
    fid_val = fid_metric.compute().item()

    # --- Save results ---
    if output_path is None:
        output_path = os.path.join(folder_path, "evaluation_results.txt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Filename\tL1\tSSIM\tLPIPS\tFID\n")
        for name, l1_val, ssim_val, lpips_val in results:
            f.write(f"{name}\t{l1_val:.6f}\t{ssim_val:.6f}\t{lpips_val:.6f}\t{fid_val:.6f}\n")

        # Averages
        avg_l1 = np.mean([r[1] for r in results])
        avg_ssim = np.mean([r[2] for r in results])
        avg_lpips = np.mean([r[3] for r in results])

        f.write(f"\nAverage L1: {avg_l1:.6f}\n")
        f.write(f"Average SSIM: {avg_ssim:.6f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.6f}\n")
        f.write(f"FID (global): {fid_val:.6f}\n")

    print(f"\n✅ Done! Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate generated vs GT images in a folder.")
    parser.add_argument("folder", type=str, help="Path to folder containing image pairs")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation txt (default: folder/evaluation_results.txt)")
    args = parser.parse_args()

    evaluate_folder(args.folder, args.output)