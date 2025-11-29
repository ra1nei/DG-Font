import os
import random, string
import torch
from PIL import Image
from datetime import datetime
from torchvision import transforms
import numpy as np
from sample import load_fontdiffuer_pipeline
from utils import save_image_with_content_style   # dùng lại utils
from tqdm import tqdm
# Nếu bạn muốn override hàm utils, bỏ comment dòng dưới và xóa import ở trên
# (nhưng theo yêu cầu thì giữ nguyên import)


# ======================
# UTILS
# ======================
def preprocess_image(path, size, device="cuda"):
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img)[None, :].to(device)


def collect_images(root_dir):
    """Thu thập toàn bộ ảnh .png, .jpg, .jpeg trong thư mục"""
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ] if os.path.exists(root_dir) else []


def save_single_image(save_dir, image, filename):
    """Lưu ảnh PIL vào thư mục"""
    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, filename))


def load_image_tensor(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img



# ======================
# NEW FUNCTION (FULL)
# ======================
def save_image_with_content_style(
    save_dir,
    gen_image_pil,
    content_image_pil=None,
    content_image_path=None,
    style_image_path=None,
    resolution=(128, 128),
    filename="out_with_cs.jpg"
):
    """
    Tạo ảnh ghép: [content | style | generated]

    Args:
        save_dir (str): thư mục lưu
        gen_image_pil (PIL.Image): ảnh generated
        content_image_pil (PIL.Image or None): ảnh content PIL nếu có
        content_image_path (str or None): đường dẫn ảnh content
        style_image_path (str): đường dẫn ảnh style
        resolution (tuple): (W, H)
        filename (str): tên file ảnh

    Returns:
        save_path (str): đường dẫn ảnh đã lưu
    """

    os.makedirs(save_dir, exist_ok=True)

    W, H = resolution

    # ----- load content -----
    if content_image_pil is not None:
        content = content_image_pil.resize((W, H))
    else:
        content = Image.open(content_image_path).convert("RGB").resize((W, H))

    # ----- load style -----
    style = Image.open(style_image_path).convert("RGB").resize((W, H))

    # ----- generated -----
    gen = gen_image_pil.resize((W, H))

    # ----- create merged canvas -----
    merged = Image.new("RGB", (W * 3, H))
    merged.paste(content, (0, 0))
    merged.paste(style, (W, 0))
    merged.paste(gen, (W * 2, 0))

    save_path = os.path.join(save_dir, filename)
    merged.save(save_path)

    return save_path




# ======================
# MAIN SAMPLING
# ======================
def batch_sampling(args):
    pipe = load_fontdiffuer_pipeline(args)
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(123)

    chinese_images = collect_images(args.chinese_dir)
    print(f"Tổng số ảnh Chinese: {len(chinese_images)}")

    samples = []
    for chi_path in chinese_images:
        font_name = os.path.basename(os.path.dirname(chi_path))
        glyph_name = os.path.splitext(os.path.basename(chi_path))[0]

        # CONTENT
        content_path = os.path.join(args.source_dir, f"{glyph_name}.png")
        # STYLE
        style_dir = os.path.join(args.english_dir, font_name)

        if args.random_style:
            style_candidates = [
                f for f in os.listdir(style_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            style_file = random.choice(style_candidates)

        else:
            style_file = "a.png"

        style_path = os.path.join(style_dir, style_file)

        if not (os.path.exists(content_path) and os.path.exists(style_path)):
            continue

        samples.append({
            "content": content_path,
            "style": style_path,
            "target": chi_path,
            "font": font_name,
            "glyph": glyph_name
        })

    print(f"Tổng số mẫu hợp lệ: {len(samples)}")

    for s in tqdm(samples, desc="🔄 Running inference", ncols=100):
        font_name, glyph_name = s["font"], s["glyph"]
        content_path, style_path, target_path = s["content"], s["style"], s["target"]

        content_img = preprocess_image(content_path, args.content_image_size, args.device)
        style_img = preprocess_image(style_path, args.style_image_size, args.device)

        with torch.no_grad():
            out_imgs = pipe.generate(
                content_images=content_img,
                style_images=style_img,
                batch_size=1,
                order=args.order,
                num_inference_step=args.num_inference_steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start,
                t_end=args.t_end,
                dm_size=args.content_image_size[0],
                algorithm_type=args.algorithm_type,
                skip_type=args.skip_type,
                method=args.method,
                correcting_x0_fn=args.correcting_x0_fn,
            )

        out_img = out_imgs[0]
        if isinstance(out_img, torch.Tensor):
            out_pil = Image.fromarray(
                ((out_img / 2 + 0.5).clamp(0, 1)
                .permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
        else:
            out_pil = out_img

        # ensure correct resolution
        out_pil = out_pil.resize(args.content_image_size)

        # Tên file như yêu cầu
        gen_filename = f"{font_name}|{glyph_name}|generated_images.png"
        gt_filename = f"{font_name}|{glyph_name}|gt_images.png"

        # Lưu ảnh generated và groundtruth
        save_single_image(args.save_dir, out_pil, gen_filename)
        target_pil = load_image_tensor(target_path, args.content_image_size)
        save_single_image(args.save_dir, target_pil, gt_filename)

        # ================================
        # Lưu ảnh ghép content-style-gen
        # ================================
        # merged_filename = f"{font_name}|{glyph_name}|merged.png"
        # save_image_with_content_style(
        #     save_dir=args.save_dir,
        #     gen_image_pil=out_pil,
        #     content_image_path=content_path,
        #     style_image_path=style_path,
        #     resolution=args.content_image_size,
        #     filename=merged_filename
        # )

    print(f"\n✅ Hoàn tất inference, ảnh lưu trong: {args.save_dir}")



# ======================
# ENTRY
# ======================
def main():
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--english_dir", type=str, required=True)
    parser.add_argument("--chinese_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/results/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--name", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--random_style", action="store_true",
                    help="Nếu bật thì chọn style random từ a-z (viết thường), và dùng file Upper+.png")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    test_img = Image.open(os.path.join(args.source_dir, os.listdir(args.source_dir)[0]))
    args.content_image_size = args.style_image_size = test_img.size
    print(f"⛏ Auto-detected image size:", args.content_image_size)

    batch_sampling(args)


if __name__ == "__main__":
    main()