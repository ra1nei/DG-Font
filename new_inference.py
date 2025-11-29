import argparse
import os
import random
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Import model definitions
# Giả sử file này đặt cùng cấp với folder models
from models.generator import Generator
from models.guidingNet import GuidingNet

def load_image(path, size):
    """Load ảnh, resize và chuẩn hóa về [-1, 1]"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    try:
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0) # Thêm batch dimension: [1, 3, H, W]
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_dir', type=str, required=True, help='Folder chứa các folder font Latin (English)')
    parser.add_argument('--content_dir', type=str, required=True, help='Folder chứa ảnh chữ Hán (Chinese/Source)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Đường dẫn file .ckpt')
    parser.add_argument('--output_dir', type=str, default='results_cross', help='Nơi lưu kết quả')
    
    parser.add_argument('--img_size', type=int, default=80)
    parser.add_argument('--sty_dim', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    # 1. Setup Device & Model
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Khởi tạo model
    C = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': 400})
    G = Generator(args.img_size, args.sty_dim, use_sn=False)

    C.to(device)
    G.to(device)
    C.eval()
    G.eval()

    # Load Checkpoint
    print(f"Loading checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        return new_state_dict

    # Load weights thông minh (ưu tiên EMA nếu có)
    c_state = checkpoint.get('C_EMA_state_dict', checkpoint.get('C_state_dict'))
    g_state = checkpoint.get('G_EMA_state_dict', checkpoint.get('G_state_dict'))
    
    C.load_state_dict(remove_module_prefix(c_state))
    G.load_state_dict(remove_module_prefix(g_state))

    # 2. Quét dữ liệu
    # Lấy danh sách các folder Font trong thư mục English (Style)
    # Cấu trúc mong đợi: args.style_dir/FontName/image.png
    style_fonts = [d for d in os.listdir(args.style_dir) if os.path.isdir(os.path.join(args.style_dir, d))]
    style_fonts.sort()
    
    # Lấy danh sách ảnh Content (Hán)
    # Hỗ trợ cả việc content_dir chứa ảnh trực tiếp hoặc chứa subfolder
    content_images = []
    for root, dirs, files in os.walk(args.content_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                content_images.append(os.path.join(root, file))
    
    print(f"Found {len(style_fonts)} Latin styles.")
    print(f"Found {len(content_images)} Chinese content glyphs.")

    if len(style_fonts) == 0 or len(content_images) == 0:
        print("Error: No data found check your paths.")
        return

    # Tạo thư mục output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 3. Vòng lặp Inference
    # Logic: Với mỗi Font Latin -> Chọn 1 ảnh ngẫu nhiên làm Style -> Áp dụng lên TẤT CẢ ảnh Hán
    
    with torch.no_grad():
        for style_font_name in tqdm(style_fonts, desc="Processing Styles"):
            style_font_path = os.path.join(args.style_dir, style_font_name)
            
            # Lấy danh sách ảnh trong folder font này
            style_imgs_list = [f for f in os.listdir(style_font_path) if f.lower().endswith(('.png', '.jpg'))]
            
            if not style_imgs_list:
                continue

            # --- KEY LOGIC: Lấy ngẫu nhiên 1 ký tự Latin làm mẫu Style ---
            random_style_img_name = random.choice(style_imgs_list)
            style_img_abs_path = os.path.join(style_font_path, random_style_img_name)
            
            # Load Style Image
            img_style_tensor = load_image(style_img_abs_path, args.img_size)
            if img_style_tensor is None: continue
            img_style_tensor = img_style_tensor.to(device)

            # Tính toán Style Code (s_ref) MỘT LẦN cho cả font này để nhanh hơn
            s_ref = C(img_style_tensor, sty=True)

            # Tạo folder kết quả cho font style này
            save_folder = os.path.join(args.output_dir, style_font_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # Duyệt qua từng ảnh Content (Hán)
            for content_path in content_images:
                img_content_tensor = load_image(content_path, args.img_size)
                if img_content_tensor is None: continue
                img_content_tensor = img_content_tensor.to(device)

                # Tính Content Code
                c_src, skip1, skip2 = C(img_content_tensor, cont=True)

                # Generator: Content Hán + Style Latin
                output, _ = G.decode(c_src, s_ref, skip1, skip2)

                # Denormalize
                output = (output + 1) / 2.0
                
                # Lưu ảnh
                # Tên file: {Tên_gốc_của_chữ_Hán}.png
                content_name = os.path.splitext(os.path.basename(content_path))[0]
                save_path = os.path.join(save_folder, f"{content_name}.png")
                save_image(output, save_path)
                
    print(f"Done! Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()