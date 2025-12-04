import os
import argparse
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

# --- IMPORTS TỪ SOURCE CODE GỐC ---
try:
    from models.generator import Generator
    from models.guidingNet import GuidingNet
except ImportError:
    print("❌ Lỗi: Không tìm thấy thư mục 'models'. Hãy đảm bảo bạn chạy script này từ thư mục gốc của dự án.")
    exit(1)

# ======================
# UTILS
# ======================

def load_image_tensor(path, size, device):
    """
    Load ảnh, resize, và chuẩn hóa về [-1, 1] cho GAN
    """
    if not os.path.exists(path):
        return None
    
    # Transform chuẩn cho GAN (thường là mean 0.5, std 0.5 để về range [-1, 1])
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        img = Image.open(path).convert("RGB")
        return tfm(img).unsqueeze(0).to(device) # Thêm batch dimension [1, C, H, W]
    except Exception as e:
        print(f"Lỗi đọc ảnh {path}: {e}")
        return None

def save_image_with_content_style(save_dir, gen_tensor, content_path, style_path, filename):
    """
    Lưu ảnh ghép: [Content | Style | Generated]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize generated tensor từ [-1, 1] về [0, 1] để lưu
    gen_tensor = (gen_tensor.clone().detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    
    # Convert sang PIL
    gen_pil = transforms.ToPILImage()(gen_tensor.squeeze(0))
    
    # Resize các ảnh khác về cùng kích thước với Gen
    W, H = gen_pil.size
    
    try:
        content_pil = Image.open(content_path).convert("RGB").resize((W, H))
        style_pil = Image.open(style_path).convert("RGB").resize((W, H))
        
        # Tạo canvas
        merged = Image.new("RGB", (W * 3, H))
        merged.paste(content_pil, (0, 0))
        merged.paste(style_pil, (W, 0))
        merged.paste(gen_pil, (W * 2, 0))
        
        save_path = os.path.join(save_dir, filename)
        merged.save(save_path)
    except Exception as e:
        print(f"Lỗi khi lưu ảnh ghép {filename}: {e}")

def load_gan_models(args, device):
    """
    Khởi tạo và load weight cho G_EMA và C_EMA
    """
    print(f"🔄 Đang tải mô hình từ: {args.checkpoint_path}")
    
    G_EMA = Generator(args.img_size, args.sty_dim, use_sn=False).to(device)
    C_EMA = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k}).to(device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    def clean_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    # Load G_EMA
    if 'G_EMA_state_dict' in checkpoint:
        G_EMA.load_state_dict(clean_state_dict(checkpoint['G_EMA_state_dict']))
    else:
        print("⚠️ Warning: Không thấy G_EMA, dùng G thường.")
        G_EMA.load_state_dict(clean_state_dict(checkpoint['G_state_dict']))

    # Load C_EMA
    if 'C_EMA_state_dict' in checkpoint:
        C_EMA.load_state_dict(clean_state_dict(checkpoint['C_EMA_state_dict']))
    else:
        print("⚠️ Warning: Không thấy C_EMA, dùng C thường.")
        C_EMA.load_state_dict(clean_state_dict(checkpoint['C_state_dict']))

    G_EMA.eval()
    C_EMA.eval()
    
    return G_EMA, C_EMA

def collect_files(root_dir):
    """Thu thập file ảnh đệ quy"""
    files = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, filename))
    return files

# ======================
# MAIN LOGIC
# ======================

def run_inference(args):
    # 1. Setup Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    print(f"⚙️ Thiết bị: {device}")

    # 2. Load Models
    try:
        G_EMA, C_EMA = load_gan_models(args, device)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    # 3. Thu thập danh sách ảnh Target (Chinese)
    print(f"📂 Đang quét thư mục target: {args.chinese_dir}")
    chinese_images = collect_files(args.chinese_dir)
    print(f"📊 Tìm thấy {len(chinese_images)} ảnh target.")

    # 4. Chuẩn bị danh sách samples (Matching logic)
    samples = []
    
    # Logic random style seed
    random.seed(42) 

    for chi_path in chinese_images:
        # Cấu trúc: .../chinese/FontName/GlyphName.png
        font_name = os.path.basename(os.path.dirname(chi_path)) # Tên Font
        glyph_name = os.path.splitext(os.path.basename(chi_path))[0] # Tên chữ (vd: 丁)

        # A. Xác định Content Path (Source)
        # Giả định source nằm trong args.source_dir/GlyphName.png
        # (Hoặc nếu source cũng chia folder thì cần sửa lại logic này tùy cấu trúc source của bạn)
        # Theo đề bài: "lấy glyph của ảnh đó trong source làm ảnh content" -> source_dir/glyph_name.png
        content_path = os.path.join(args.source_dir, f"{glyph_name}.png")
        
        # Fallback nếu trong source nó nằm trong folder con (ví dụ source/A+/glyph.png)
        if not os.path.exists(content_path):
             # Thử tìm đệ quy hoặc giả định một cấu trúc khác nếu cần. 
             # Hiện tại giữ simple: source_dir/glyph.png
             pass

        # B. Xác định Style Path (English)
        # Style nằm trong: english_dir/FontName/...
        style_dir = os.path.join(args.english_dir, font_name)
        
        if not os.path.exists(style_dir):
            continue # Không có folder style tương ứng font này -> Skip

        # Logic chọn file Style (Random vs Fixed)
        style_file = None
        
        if args.random_style:
            # Lấy danh sách ảnh trong folder style đó
            candidates = [f for f in os.listdir(style_dir) if f.lower().endswith(('.png', '.jpg'))]
            
            # Lọc theo mode nếu cần (ví dụ chỉ lấy chữ hoa)
            if args.random_mode == "upper":
                # Lọc thô sơ: Tên file dài 1 ký tự và là chữ hoa (A.png) hoặc A+.png
                candidates = [f for f in candidates if f[0].isupper()]
            
            if candidates:
                style_file = random.choice(candidates)
        else:
            # Fixed style (ví dụ chọn 'A+.png' hoặc 'a.png')
            # Thử tìm file chính xác
            possible_names = [args.fixed_style, args.fixed_style + ".png", args.fixed_style + ".jpg"]
            for name in possible_names:
                if os.path.exists(os.path.join(style_dir, name)):
                    style_file = name
                    break
        
        if style_file:
            style_path = os.path.join(style_dir, style_file)
            
            # Chỉ thêm vào list nếu cả Content và Style đều tồn tại
            if os.path.exists(content_path) and os.path.exists(style_path):
                samples.append({
                    "content": content_path,
                    "style": style_path,
                    "target": chi_path,
                    "font_name": font_name,
                    "glyph_name": glyph_name
                })

    print(f"✅ Đã ghép cặp thành công: {len(samples)} mẫu.")

    # 5. Chạy Inference Loop
    os.makedirs(args.save_dir, exist_ok=True)
    
    with torch.no_grad():
        for s in tqdm(samples, desc="🚀 Running Inference", ncols=100):
            # Load Tensors
            c_img = load_image_tensor(s["content"], args.img_size, device)
            s_img = load_image_tensor(s["style"], args.img_size, device)
            
            if c_img is None or s_img is None:
                continue

            # --- GAN FORWARD PASS ---
            # 1. Extract Content
            c_code, skip1, skip2 = G_EMA.cnt_encoder(c_img)
            # 2. Extract Style
            s_code = C_EMA(s_img, sty=True)
            # 3. Decode / Generate
            fake_img, _ = G_EMA.decode(c_code, s_code, skip1, skip2)
            # ------------------------

            # Save Results
            # Tên file: Font_Glyph_Generated.png
            safe_glyph = "".join([c if c.isalnum() else "_" for c in s["glyph_name"]]) # Xử lý ký tự đặc biệt
            base_name = f"{s['font_name']}_{safe_glyph}"
            
            normalized_fake_img = (fake_img.clone().detach() * 0.5 + 0.5).clamp(0, 1)
            
            # 1. Lưu ảnh lẻ (Generated)
            # Dùng normalize=False vì ảnh đã được chuẩn hóa thủ công
            vutils.save_image(
                normalized_fake_img, 
                os.path.join(args.save_dir, f"{base_name}_gen.png"),
                normalize=False,  # Bỏ normalize=True và tham số range
            )

            # 2. Lưu ảnh ghép (Content | Style | Gen) - Hàm này đã được sửa bên trong
            save_image_with_content_style(
                save_dir=os.path.join(args.save_dir, "merged_view"),
                gen_tensor=fake_img, # Vẫn truyền tensor [-1, 1] vì hàm save_image_with_content_style xử lý
                content_path=s["content"],
                style_path=s["style"],
                filename=f"{base_name}_merged.jpg"
            )

    print(f"\n🎉 Hoàn tất! Kết quả lưu tại: {args.save_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference GAN Font Generation")
    
    # Paths
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--source_dir', type=str, required=True, help='Folder chứa ảnh Content gốc (Source)')
    parser.add_argument('--chinese_dir', type=str, required=True, help='Folder chứa ảnh Target (Chinese) - Dùng để duyệt danh sách')
    parser.add_argument('--english_dir', type=str, required=True, help='Folder chứa ảnh Style (English)')
    parser.add_argument('--save_dir', type=str, default='./results', help='Folder lưu kết quả')

    # Style Logic
    parser.add_argument("--random_style", action="store_true", help="Chọn style ngẫu nhiên từ folder English")
    parser.add_argument("--random_mode", type=str, default="full", choices=["full", "upper"], help="Chế độ random")
    parser.add_argument("--fixed_style", type=str, default="A", help="Tên file style cố định (VD: A, A+, a) nếu không dùng random")

    # Model Params
    parser.add_argument('--img_size', type=int, default=80, help='Kích thước ảnh model (default: 80)')
    parser.add_argument('--sty_dim', type=int, default=128, help='Style vector dimension')
    parser.add_argument('--output_k', type=int, default=400, help='Số class output của GuidingNet')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_inference(args)