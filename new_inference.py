import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from models.generator import Generator
from models.guidingNet import GuidingNet

def load_image(path, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0) # Thêm batch dimension

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str, required=True, help='Path to content image (chữ nội dung)')
    parser.add_argument('--style_path', type=str, required=True, help='Path to style image (font mẫu)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to .ckpt file (File model cụ thể)')
    parser.add_argument('--output_path', type=str, default='result.png', help='Path to save result')
    parser.add_argument('--img_size', type=int, default=80)
    parser.add_argument('--sty_dim', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 1. Khởi tạo Model
    # Code này dùng GuidingNet cho Content Encoder và Style Encoder
    # Và Generator cho Decoder
    C = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': 400}) # disc num k quan trọng lúc infer
    G = Generator(args.img_size, args.sty_dim, use_sn=False)

    C.to(device)
    G.to(device)
    C.eval()
    G.eval()

    # 2. Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    # Xử lý key 'module.' nếu train bằng DataParallel
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    # Load C (Context Encoder + Style Encoder)
    if 'C_EMA_state_dict' in checkpoint:
        C.load_state_dict(remove_module_prefix(checkpoint['C_EMA_state_dict']))
    else:
        C.load_state_dict(remove_module_prefix(checkpoint['C_state_dict']))

    # Load G (Decoder)
    if 'G_EMA_state_dict' in checkpoint:
        G.load_state_dict(remove_module_prefix(checkpoint['G_EMA_state_dict']))
    else:
        G.load_state_dict(remove_module_prefix(checkpoint['G_state_dict']))

    # 3. Inference
    print("Processing...")
    with torch.no_grad():
        # Load ảnh
        img_content = load_image(args.content_path, args.img_size).to(device)
        img_style = load_image(args.style_path, args.img_size).to(device)

        # Encode Content
        # C trả về: content_code, skip1, skip2
        c_src, skip1, skip2 = C(img_content, cont=True)
        
        # Encode Style
        # C trả về: style_code
        s_ref = C(img_style, sty=True)

        # Decode (Generate)
        output, _ = G.decode(c_src, s_ref, skip1, skip2)

        # Denormalize để lưu ảnh cho đẹp
        output = (output + 1) / 2.0

        save_image(output, args.output_path)
        print(f"Saved result to {args.output_path}")

if __name__ == '__main__':
    main()