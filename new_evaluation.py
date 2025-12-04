import os
import torch
import lpips
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
import warnings

# Tắt cảnh báo SSIM (thường xảy ra khi so sánh tensor)
warnings.filterwarnings("ignore", category=UserWarning)


# --- Utility functions ---
def load_image(path):
    """
    Tải ảnh, chuyển sang tensor và chuẩn hóa về dải [0, 1]
    """
    try:
        img = Image.open(path).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(img).unsqueeze(0) # [1, C, H, W], dải [0, 1]
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Lỗi khi tải ảnh {path}: {e}")
        return None


def l1_loss(img1, img2):
    """Tính L1 Loss giữa hai tensor [0, 1]"""
    return torch.mean(torch.abs(img1 - img2)).item()


def ssim_score(img1, img2):
    """
    Tính Structural Similarity Index (SSIM)
    Yêu cầu tensor đã được chuyển sang CPU và dải [0, 1]
    """
    # SSIM cần tensor ở dạng numpy (H, W, C) hoặc (C, H, W)
    # Ta chuyển từ (1, C, H, W) -> (H, W, C) numpy
    img1_np = img1.squeeze().permute(1, 2, 0).numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).numpy()
    
    # Sử dụng channel_axis=2 cho định dạng (H, W, C) và data_range=1.0 cho ảnh [0, 1]
    return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)


# --- Main evaluation ---
def evaluate_folder(root_folder_path, output_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    # 1. Xác định đường dẫn thư mục con
    gen_folder = os.path.join(root_folder_path, "generated_images")
    gt_folder = os.path.join(root_folder_path, "gt_images")

    if not os.path.isdir(gen_folder) or not os.path.isdir(gt_folder):
        print(f"❌ Lỗi: Không tìm thấy thư mục 'generated_images' hoặc 'gt_images' trong {root_folder_path}")
        print("Vui lòng đảm bảo cấu trúc thư mục là: root_folder_path/generated_images và root_folder_path/gt_images")
        return

    # 2. Lấy danh sách ảnh sinh ra
    generated_files = os.listdir(gen_folder)
    
    # Lọc chỉ lấy các file .png hoặc .jpg
    image_extensions = ('.png', '.jpg', '.jpeg')
    generated_files = [f for f in generated_files if f.lower().endswith(image_extensions)]
    
    print(f"📊 Tìm thấy {len(generated_files)} ảnh sinh ra để đánh giá.")

    # 3. Setup models
    try:
        lpips_model = lpips.LPIPS(net='vgg').to(device)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo LPIPS: {e}. Vui lòng đảm bảo đã cài đặt lpips (pip install lpips) và torchvision.")
        return
        
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device) # normalize=True cho tensor [0, 1]
    
    results = []
    gen_fid_updates = []
    gt_fid_updates = []

    # 4. Duyệt và đánh giá từng cặp ảnh
    for gen_file in tqdm(generated_files, desc="🔄 Đang đánh giá"):
        # Tên file sinh ra: Font_Glyph_gen.png
        # Tên file GT tương ứng: Font_Glyph_gt.png
        
        # Tạo tên file GT dựa trên tên file Generated
        base_name_without_suffix = gen_file.rsplit('_', 1)[0] # Bỏ _gen.png hoặc _gen.jpg
        gt_file = f"{base_name_without_suffix}_gt.png"

        gen_path = os.path.join(gen_folder, gen_file)
        gt_path = os.path.join(gt_folder, gt_file)
        
        # Xử lý trường hợp file Generated không có đuôi .png
        if not os.path.exists(gt_path):
            gt_file = f"{base_name_without_suffix}_gt.jpg"
            gt_path = os.path.join(gt_folder, gt_file)
            
            if not os.path.exists(gt_path):
                # Thử tìm file GT với tên file Generated y hệt (nếu có lỗi logic tên file)
                # Đây là fallback nếu logic đặt tên file không chuẩn
                # gt_path_fallback = os.path.join(gt_folder, gen_file.replace("_gen", "_gt"))
                # if os.path.exists(gt_path_fallback):
                #     gt_path = gt_path_fallback
                # else:
                print(f"⚠️ Missing GT file cho {gen_file}. Đã bỏ qua.")
                continue

        # Tải ảnh (ảnh đã ở dải [0, 1])
        gen_img = load_image(gen_path)
        gt_img = load_image(gt_path)

        if gen_img is None or gt_img is None:
            continue
            
        gen_img = gen_img.to(device)
        gt_img = gt_img.to(device)

        # Kiểm tra kích thước tensor trước khi tính toán
        if gen_img.shape != gt_img.shape:
             print(f"❌ Bỏ qua cặp {gen_file} và {gt_file}: Kích thước tensor khác nhau ({gen_img.shape} vs {gt_img.shape})")
             continue

        # --- Per-image metrics ---
        l1_val = l1_loss(gen_img, gt_img)
        ssim_val = ssim_score(gen_img.cpu(), gt_img.cpu())
        
        # LPIPS yêu cầu ảnh được chuẩn hóa về [-1, 1], nhưng lpips.LPIPS(net='vgg') 
        # thường xử lý chuẩn hóa nội bộ từ [0, 1] sang [-1, 1] khi dùng VGG/AlexNet.
        # Nếu gặp lỗi, ta có thể phải chuẩn hóa thủ công.
        try:
            lpips_val = lpips_model(gen_img, gt_img).item()
        except RuntimeError:
            # Fallback: Chuẩn hóa thủ công cho LPIPS (chuyển [0, 1] -> [-1, 1])
            gen_lpips = gen_img * 2 - 1
            gt_lpips = gt_img * 2 - 1
            lpips_val = lpips_model(gen_lpips, gt_lpips).item()


        results.append((base_name_without_suffix, l1_val, ssim_val, lpips_val))
        
        # --- Store for FID (global) ---
        # FID metric cần input là torch.uint8 (0-255)
        # Chuyển tensor [0, 1] sang tensor [0, 255] (uint8)
        gen_fid_updates.append((gen_img * 255).byte())
        gt_fid_updates.append((gt_img * 255).byte())

    # 5. Compute FID (global)
    if not gen_fid_updates:
        print("Không có cặp ảnh hợp lệ để tính toán. Kết thúc.")
        return
        
    print(f"\nGen: {len(gen_fid_updates)} ảnh | GT: {len(gt_fid_updates)} ảnh")

    # Cập nhật metric FID
    for img in gen_fid_updates:
        fid_metric.update(img, real=False)
    for img in gt_fid_updates:
        fid_metric.update(img, real=True)

    try:
        fid_val = fid_metric.compute().item()
    except Exception as e:
        print(f"❌ Lỗi khi tính FID: {e}. Đảm bảo số lượng ảnh đủ (>1) và kích thước là 299x299 cho InceptionV3.")
        fid_val = float('nan')
        

    # 6. Save results
    if output_path is None:
        output_path = os.path.join(root_folder_path, "evaluation_results.txt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    avg_l1 = np.mean([r[1] for r in results])
    avg_ssim = np.mean([r[2] for r in results])
    avg_lpips = np.mean([r[3] for r in results])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("--- TÓM TẮT KẾT QUẢ ĐÁNH GIÁ ---\n")
        f.write(f"Tổng số cặp ảnh hợp lệ: {len(results)}\n\n")
        f.write(f"Average L1: {avg_l1:.6f}\n")
        f.write(f"Average SSIM: {avg_ssim:.6f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.6f}\n")
        f.write(f"FID (global): {fid_val:.6f}\n\n")
        
        f.write("--- KẾT QUẢ CHI TIẾT THEO TỪNG CẶP ẢNH ---\n")
        f.write("Filename\tL1\tSSIM\tLPIPS\n")
        for name, l1_val, ssim_val, lpips_val in results:
            f.write(f"{name}\t{l1_val:.6f}\t{ssim_val:.6f}\t{lpips_val:.6f}\n")


    print(f"\n✅ Done! Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated vs GT images in a folder.")
    parser.add_argument("folder", type=str, help="Path to the ROOT folder containing 'generated_images' and 'gt_images' subfolders (e.g., ./results/unknown_content)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation txt (default: folder/evaluation_results.txt)")
    args = parser.parse_args()

    evaluate_folder(args.folder, args.output)