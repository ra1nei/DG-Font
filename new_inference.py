import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from collections import OrderedDict

# --- IMPORTS TỪ SOURCE CODE GỐC ---
# Giả định file này nằm cùng thư mục gốc với main.py để có thể truy cập folder 'models'
try:
    from models.generator import Generator
    from models.guidingNet import GuidingNet
except ImportError:
    print("Lỗi: Không tìm thấy thư mục 'models'. Hãy đảm bảo bạn chạy script này từ thư mục gốc của dự án (nơi chứa main.py và thư mục models).")
    exit(1)

# --- HELPER FUNCTIONS ---

def load_image(image_path, img_size, device):
    """
    Tải và tiền xử lý hình ảnh đầu vào.
    Chuyển đổi sang Grayscale nếu mô hình yêu cầu (thường font là grayscale), 
    nhưng based on main.py channels seems implied standard. 
    Code gốc main.py dùng datasetgetter, thường trả về tensor chuẩn hóa.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

    # Transform giống như quy trình training thường dùng cho GAN
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # Thêm batch dimension
    return image_tensor

def load_models(args, device):
    """
    Khởi tạo và tải trọng các mô hình G_EMA và C_EMA dựa trên cấu trúc trong main.py.
    """
    print(f"Đang khởi tạo các mô hình trên device: {device}")
    
    # 1. Khởi tạo Mô hình (Dựa trên main.py build_model)
    # G: Generator
    # C: GuidingNet
    
    # Cấu hình từ main.py:
    # networks['G'] = Generator(args.img_size, args.sty_dim, use_sn=False)
    # networks['C'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
    
    G_EMA = Generator(args.img_size, args.sty_dim, use_sn=False).to(device)
    C_EMA = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k}).to(device)

    # 2. Tải Checkpoint
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {args.checkpoint_path}")
        
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Hàm helper để tải state_dict (xử lý 'module.' prefix nếu train bằng DataParallel)
    def load_state_dict_safe(model, state_dict_key):
        if state_dict_key not in checkpoint:
            print(f"Warning: Key {state_dict_key} not found in checkpoint.")
            return False
            
        state_dict = checkpoint[state_dict_key]
        new_state_dict = OrderedDict()
        
        # Kiểm tra xem có cần loại bỏ prefix 'module.' không
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        return True

    # 3. Tải trọng số cho G_EMA
    # Trong main.py save_model, key là 'G_EMA_state_dict'
    if not load_state_dict_safe(G_EMA, 'G_EMA_state_dict'):
        print("Đang thử tải trọng số G thường (không phải EMA)...")
        load_state_dict_safe(G_EMA, 'G_state_dict')

    # 4. Tải trọng số cho C_EMA
    # Trong main.py save_model, key là 'C_EMA_state_dict' (lưu ý: main.py skip saving EMA cho C/G nếu là distributed?? 
    # Check lại main.py: save_model skip G_EMA, C_EMA? 
    # Dòng 377 main.py: if name in ['G_EMA', 'C_EMA']: continue. 
    # NHƯNG validation.py dùng G_EMA. Vậy checkpoint load từ đâu?
    # À, main.py dòng 330 khởi tạo G_EMA. 
    # Nếu checkpoint không lưu G_EMA riêng, ta load từ G_state_dict.
    
    if not load_state_dict_safe(C_EMA, 'C_EMA_state_dict'):
        print("Không tìm thấy C_EMA, tải từ C_state_dict...")
        load_state_dict_safe(C_EMA, 'C_state_dict')

    # Đảm bảo các mô hình ở chế độ đánh giá
    G_EMA.eval()
    C_EMA.eval()
    
    return G_EMA, C_EMA

def run_inference(args):
    """
    Thực hiện toàn bộ quá trình inference: tách Content, trích xuất Style và tổng hợp.
    """
    # 1. Cấu hình Device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    
    print(f"Thiết bị được chọn: {device}")

    # 2. Tải Mô hình
    try:
        G_EMA, C_EMA = load_models(args, device)
    except Exception as e:
        print(f"Lỗi xảy ra khi tải/khởi tạo mô hình: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Tải Hình ảnh
    try:
        x_src = load_image(args.source_image, args.img_size, device)
        x_ref = load_image(args.reference_image, args.img_size, device)
    except FileNotFoundError as e:
        print(f"Lỗi tải hình ảnh: {e}")
        return
    
    # 4. Thực hiện Inference (Tạo ảnh)
    print("Bắt đầu quá trình inference...")
    with torch.no_grad():
        # Logic này khớp với validation.py dòng 102-104
        
        # A. Trích xuất Content từ ảnh nguồn (x_src)
        # G_EMA.cnt_encoder trả về: c_src, skip1, skip2
        c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src)
        
        # B. Trích xuất Style từ ảnh tham chiếu (x_ref)
        # C_EMA(..., sty=True) trả về s_ref
        # Lưu ý: validation.py dùng x_ref_tmp lặp lại batch, ở đây ta inference 1 ảnh nên không cần repeat nếu batch=1
        s_ref = C_EMA(x_ref, sty=True)
        
        # C. Tổng hợp và Giải mã
        # G_EMA.decode trả về x_res_ema_tmp, _
        x_res, _ = G_EMA.decode(c_src, s_ref, skip1, skip2)

    print("Inference hoàn tất. Chuẩn bị lưu kết quả.")
    
    # 5. Lưu kết quả
    # Resize về dạng hiển thị tốt
    output_grid = torch.cat([x_src, x_ref, x_res], dim=0)
    
    try:
        vutils.save_image(
            output_grid.cpu(), 
            args.output_path, 
            normalize=True, 
            # nrow=3 để xếp ngang: Source | Reference | Result
            nrow=3 
        )
        print(f"Đã lưu kết quả thành công tại: {args.output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu ảnh: {e}")

def parse_args():
    """
    Định nghĩa và phân tích các đối số dòng lệnh.
    """
    parser = argparse.ArgumentParser(description="Inference Module for Content-Style Transfer Model")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Đường dẫn đến file checkpoint (.pth) (VD: ./logs/model/checkpoint.pth)')
    parser.add_argument('--source_image', type=str, required=True,
                        help='Đường dẫn đến ảnh nguồn (Content/Ký tự).')
    parser.add_argument('--reference_image', type=str, required=True,
                        help='Đường dẫn đến ảnh tham chiếu (Style/Font).')
    parser.add_argument('--output_path', type=str, default='result.jpg',
                        help='Đường dẫn để lưu ảnh kết quả.')
    
    # Các tham số mặc định khớp với main.py
    parser.add_argument('--img_size', type=int, default=80, 
                        help='Kích thước ảnh đầu vào (main.py default=80).')
    parser.add_argument('--sty_dim', type=int, default=128,
                        help='Kích thước vector style (main.py default=128).')
    parser.add_argument('--output_k', type=int, default=400,
                        help='Tổng số lớp/styles (main.py default=400).')
    
    parser.add_argument('--gpu', type=int, default=0,
                        help='Chỉ mục GPU để sử dụng.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_inference(args)