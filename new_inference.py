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
    # Fallback giả định nếu không chạy trong project thật để tránh lỗi import khi review code
    print("⚠️ Cảnh báo: Không import được models. Chỉ chạy được nếu cấu trúc thư mục đúng.")
    Generator = None
    GuidingNet = None

# ======================
# UTILS (Giữ nguyên phần xử lý ảnh tốt của bạn)
# ======================

def load_image_tensor(path, size, device, normalize=True):
    if not os.path.exists(path): return None
    transform_list = [transforms.Resize((size, size)), transforms.ToTensor()]
    if normalize: transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    tfm = transforms.Compose(transform_list)
    try:
        img = Image.open(path).convert("RGB")
        return tfm(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Lỗi đọc ảnh {path}: {e}")
        return None

def save_image_with_content_style(save_dir, gen_tensor, content_path, style_path, filename):
    os.makedirs(save_dir, exist_ok=True)
    gen_tensor_norm = (gen_tensor.cpu() * 0.5 + 0.5).clamp(0, 1)
    gen_pil = transforms.ToPILImage()(gen_tensor_norm.squeeze(0))
    W, H = gen_pil.size
    try:
        content_pil = Image.open(content_path).convert("RGB").resize((W, H))
        style_pil = Image.open(style_path).convert("RGB").resize((W, H))
        merged = Image.new("RGB", (W * 3, H))
        merged.paste(content_pil, (0, 0))
        merged.paste(style_pil, (W, 0))
        merged.paste(gen_pil, (W * 2, 0))
        merged.save(os.path.join(save_dir, filename))
    except Exception as e:
        print(f"Lỗi lưu ảnh ghép {filename}: {e}")

def load_gan_models(args, device):
    print(f"🔄 Đang tải mô hình từ: {args.checkpoint_path}")
    if Generator is None: return None, None # Safety check

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

    # Load Weights (Ưu tiên EMA)
    if 'G_EMA_state_dict' in checkpoint:
        G_EMA.load_state_dict(clean_state_dict(checkpoint['G_EMA_state_dict']))
    else:
        G_EMA.load_state_dict(clean_state_dict(checkpoint['G_state_dict']))

    if 'C_EMA_state_dict' in checkpoint:
        C_EMA.load_state_dict(clean_state_dict(checkpoint['C_EMA_state_dict']))
    else:
        C_EMA.load_state_dict(clean_state_dict(checkpoint['C_state_dict']))

    G_EMA.eval()
    C_EMA.eval()
    return G_EMA, C_EMA

def collect_files(root_dir):
    if not os.path.exists(root_dir): return []
    return [os.path.join(r, f) for r, _, fs in os.walk(root_dir) for f in fs if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ======================
# MAIN LOGIC (NÂNG CẤP)
# ======================

def run_inference(args):
    # 1. Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    print(f"⚙️ Thiết bị: {device} | Mode: {args.direction}")
    
    G_EMA, C_EMA = load_gan_models(args, device)
    if G_EMA is None: return

    # Tạo folder output
    gen_dir = os.path.join(args.save_dir, 'generated_images')
    gt_dir = os.path.join(args.save_dir, 'gt_images')
    merged_dir = os.path.join(args.save_dir, 'merged_view')
    for d in [gen_dir, gt_dir, merged_dir]: os.makedirs(d, exist_ok=True)

    samples = []
    random.seed(42)

    # ==========================================
    # LOGIC C2E (Chinese Style -> English Content)
    # ==========================================
    if args.direction == "c2e":
        print(f"🚀 C2E Config | Phase: {args.phase} | Complexity: {args.complexity}")
        
        # 1. Targets là ảnh tiếng Anh (English GT)
        target_images = collect_files(args.english_dir)
        print(f"📊 Tìm thấy {len(target_images)} ảnh English targets.")

        # 2. Xác định nguồn tìm Style (Train hay Test Chinese)
        style_search_root = args.chinese_dir
        if args.phase == "test_unknown_content":
            if not args.chinese_train_dir:
                raise ValueError("❌ Cần --chinese_train_dir cho test_unknown_content")
            print(f"🔄 Switch Style Source -> Train Dir: {args.chinese_train_dir}")
            style_search_root = args.chinese_train_dir
        
        # 3. Tạo tập mẫu (Reference Complexity Set)
        candidate_glyph_set = set()
        if args.complexity == "all":
            try:
                first_font = os.listdir(style_search_root)[0]
                candidate_glyph_set = set(os.listdir(os.path.join(style_search_root, first_font)))
            except: pass
        else:
            # Load từ folder complexity đã phân loại
            comp_map = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
            ref_folder = os.path.join(args.complexity_root, comp_map[args.complexity])
            if os.path.exists(ref_folder):
                candidate_glyph_set = set([f for f in os.listdir(ref_folder) if f.endswith(('.png', '.jpg'))])
        
        print(f"🔍 Tìm thấy {len(candidate_glyph_set)} glyph mẫu cho độ khó '{args.complexity}'")
        if not candidate_glyph_set: return

        # 4. Cache và Matching
        font_cache = {} # Cache danh sách file của font để tránh os.listdir nhiều lần

        for eng_path in target_images:
            font_name = os.path.basename(os.path.dirname(eng_path))
            glyph_name = os.path.splitext(os.path.basename(eng_path))[0]

            # Rule: Bỏ qua chữ in hoa cho C2E
            if glyph_name.isupper(): continue

            # Content: Ảnh skeleton tiếng Anh
            content_path = os.path.join(args.source_dir, f"{glyph_name}.png")
            
            # Style: Tìm trong folder Chinese tương ứng
            chinese_font_dir = os.path.join(style_search_root, font_name)
            if not os.path.exists(chinese_font_dir): continue

            # Intersection Logic + Cache
            if font_name not in font_cache:
                try:
                    actual_files = set(os.listdir(chinese_font_dir))
                    valid = list(candidate_glyph_set.intersection(actual_files))
                    font_cache[font_name] = valid
                except: font_cache[font_name] = []
            
            valid_candidates = font_cache[font_name]
            if not valid_candidates: continue

            # Chọn ngẫu nhiên 1 file chắc chắn tồn tại
            style_path = os.path.join(chinese_font_dir, random.choice(valid_candidates))

            if os.path.exists(content_path):
                samples.append({
                    "content": content_path, "style": style_path, "target": eng_path,
                    "font": font_name, "glyph": glyph_name
                })

    # ==========================================
    # LOGIC E2C (English Style -> Chinese Content)
    # ==========================================
    elif args.direction == "e2c":
        print("🚀 E2C Config: English Style -> Chinese Content")
        target_images = collect_files(args.chinese_dir) # Target là Chinese GT
        
        for chi_path in target_images:
            font_name = os.path.basename(os.path.dirname(chi_path))
            glyph_name = os.path.splitext(os.path.basename(chi_path))[0]
            
            # Content: Ảnh skeleton Chinese
            content_path = os.path.join(args.source_dir, f"{glyph_name}.png")
            # Style: Folder English
            style_dir = os.path.join(args.english_dir, font_name)
            
            if not os.path.exists(style_dir): continue

            # Simple Random selection (như code cũ của bạn, nhưng gọn hơn)
            style_file = None
            if args.random_style:
                cands = [f for f in os.listdir(style_dir) if f.endswith('.png')]
                if args.random_mode == "upper": 
                    cands = [f for f in cands if f[0].isupper()]
                if cands: style_file = random.choice(cands)
            else:
                if os.path.exists(os.path.join(style_dir, "A.png")): style_file = "A.png"
            
            if style_file and os.path.exists(content_path):
                samples.append({
                    "content": content_path, "style": os.path.join(style_dir, style_file), "target": chi_path,
                    "font": font_name, "glyph": glyph_name
                })

    print(f"✅ Đã ghép cặp thành công: {len(samples)} mẫu.")

    # 5. INFERENCE LOOP
    with torch.no_grad():
        for s in tqdm(samples, desc="Processing", ncols=100):
            # Load
            c_img = load_image_tensor(s["content"], args.img_size, device)
            s_img = load_image_tensor(s["style"], args.img_size, device)
            
            if c_img is None or s_img is None: continue

            # Forward
            c_code, skip1, skip2 = G_EMA.cnt_encoder(c_img)
            s_code = C_EMA(s_img, sty=True)
            fake_img, _ = G_EMA.decode(c_code, s_code, skip1, skip2)

            # Save
            safe_glyph = "".join([c if c.isalnum() else "_" for c in s["glyph_name"]])
            base_name = f"{s['font_name']}_{safe_glyph}"
            
            # Save Gen (Normalized)
            gen_norm = (fake_img.clone().detach() * 0.5 + 0.5).clamp(0, 1)
            vutils.save_image(gen_norm, os.path.join(gen_dir, f"{base_name}_gen.png"), normalize=False)
            
            # Save GT (Normalized)
            gt_tensor = load_image_tensor(s["target"], args.img_size, device, normalize=False)
            if gt_tensor is not None:
                vutils.save_image(gt_tensor, os.path.join(gt_dir, f"{base_name}_gt.png"), normalize=False)

            # Save Merged
            save_image_with_content_style(merged_dir, fake_img, s["content"], s["style"], f"{base_name}_merged.jpg")

    print(f"\n🎉 Done! Results at: {args.save_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    # Path Basics
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--source_dir', type=str, required=True, help="Folder content skeleton")
    parser.add_argument('--save_dir', type=str, default='./results')
    
    # Dataset Paths
    parser.add_argument('--english_dir', type=str, required=True, help="Dataset English")
    parser.add_argument('--chinese_dir', type=str, required=True, help="Dataset Chinese (Test)")
    parser.add_argument('--chinese_train_dir', type=str, default=None, help="Dataset Chinese (Train) for unknown content")
    
    # Logic configs
    parser.add_argument('--direction', type=str, default='c2e', choices=['c2e', 'e2c'])
    parser.add_argument('--phase', type=str, default='test_unknown_style', choices=['test_unknown_content', 'test_unknown_style'])
    parser.add_argument('--complexity', type=str, default='hard', choices=['all', 'easy', 'medium', 'hard'])
    parser.add_argument('--complexity_root', type=str, default=None, help="Folder containing Easy/Medium/Hard subfolders")

    # Style options (Only for E2C mainly)
    parser.add_argument("--random_style", action="store_true")
    parser.add_argument("--random_mode", type=str, default="full")

    # Model Params
    parser.add_argument('--img_size', type=int, default=80)
    parser.add_argument('--sty_dim', type=int, default=128)
    parser.add_argument('--output_k', type=int, default=400)
    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_inference(args)