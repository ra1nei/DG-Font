import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torch.nn.functional as F

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from scipy import linalg
from tools.utils import *


def validateUN(data_loader, networks, epoch, args, additional=None):
    # set nets
    D = networks['D']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module

    # switch to train mode
    D.eval()
    G.eval()
    C.eval()
    C_EMA.eval()
    G_EMA.eval()

    # data loader
    val_dataset = data_loader['TRAINSET']
    val_loader = data_loader['VAL']

    # Danh sách lưu trữ tensor hình ảnh đã tải cho mỗi lớp
    x_each_cls = []

    # Danh sách chỉ mục lớp hợp lệ được tải thành công
    valid_cls_indices = []

    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):

            # 1. Robust selection
            tmp_cls_set_all = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()
            num_available = len(tmp_cls_set_all)

            # Số lượng mẫu thực tế để lấy
            num_to_take = min(num_available, args.val_num)

            if num_to_take == 0:
                print(f"Warning: Skipping class index {cls_idx} "
                      f"(target: {args.att_to_use[cls_idx]}) due to no images found.")
                continue

            # Lấy mẫu
            tmp_cls_set = tmp_cls_set_all[-num_to_take:]
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set.flatten())

            tmp_dl = torch.utils.data.DataLoader(
                tmp_ds, batch_size=args.val_num, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False
            )
            tmp_iter = iter(tmp_dl)
            tmp_sample = None

            for sample_idx in range(len(tmp_iter)):
                imgs, _ = next(tmp_iter)
                x_ = imgs
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)

            if tmp_sample is not None:
                x_each_cls.append(tmp_sample)
                valid_cls_indices.append(cls_idx)

    # 3. Kiểm tra trước FID
    if not x_each_cls:
        print("Warning: No valid class data loaded for validation. Returning.")
        return

    valid_att_to_use = [args.att_to_use[i] for i in valid_cls_indices]

    if epoch >= args.fid_start:
        # Reference guided
        with torch.no_grad():
            ones = torch.ones(
                1, x_each_cls[0].size(1),
                x_each_cls[0].size(2),
                x_each_cls[0].size(3)
            ).cuda(args.gpu, non_blocking=True)

            for src_idx in range(len(x_each_cls)):
                x_src = x_each_cls[src_idx][:args.val_batch].cuda(args.gpu, non_blocking=True)
                rnd_idx = torch.randperm(x_each_cls[src_idx].size(0))[:args.val_batch]
                x_src_rnd = x_each_cls[src_idx][rnd_idx].cuda(args.gpu, non_blocking=True)

                for ref_idx in range(len(x_each_cls)):
                    x_res_ema = torch.cat((ones, x_src), 0)
                    x_rnd_ema = torch.cat((ones, x_src_rnd), 0)

                    x_ref = x_each_cls[ref_idx][:args.val_batch].cuda(args.gpu, non_blocking=True)
                    rnd_idx = torch.randperm(x_each_cls[ref_idx].size(0))[:args.val_batch]
                    x_ref_rnd = x_each_cls[ref_idx][rnd_idx].cuda(args.gpu, non_blocking=True)

                    for sample_idx in range(args.val_batch):
                        x_ref_tmp = x_ref[sample_idx: sample_idx + 1].repeat((args.val_batch, 1, 1, 1))

                        c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src)
                        s_ref = C_EMA(x_ref_tmp, sty=True)
                        x_res_ema_tmp, _ = G_EMA.decode(c_src, s_ref, skip1, skip2)

                        x_ref_tmp = x_ref_rnd[sample_idx: sample_idx + 1].repeat((args.val_batch, 1, 1, 1))

                        c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src_rnd)
                        s_ref = C_EMA(x_ref_tmp, sty=True)
                        x_rnd_ema_tmp, _ = G_EMA.decode(c_src, s_ref, skip1, skip2)

                        x_res_ema_tmp = torch.cat(
                            (x_ref[sample_idx: sample_idx + 1], x_res_ema_tmp), 0
                        )
                        x_res_ema = torch.cat((x_res_ema, x_res_ema_tmp), 0)

                        x_rnd_ema_tmp = torch.cat(
                            (x_ref_rnd[sample_idx: sample_idx + 1], x_rnd_ema_tmp), 0
                        )
                        x_rnd_ema = torch.cat((x_rnd_ema, x_rnd_ema_tmp), 0)

                    src_cls_original_idx = valid_cls_indices[src_idx]
                    ref_cls_original_idx = valid_cls_indices[ref_idx]

                    vutils.save_image(
                        x_res_ema,
                        os.path.join(
                            args.res_dir,
                            '{}_EMA_{}_{}{}.jpg'.format(
                                args.gpu, epoch + 1,
                                src_cls_original_idx,
                                ref_cls_original_idx
                            )
                        ),
                        normalize=True,
                        nrow=(x_res_ema.size(0) // (x_src.size(0) + 2) + 1)
                    )
                    vutils.save_image(
                        x_rnd_ema,
                        os.path.join(
                            args.res_dir,
                            '{}_RNDEMA_{}_{}{}.jpg'.format(
                                args.gpu, epoch + 1,
                                src_cls_original_idx,
                                ref_cls_original_idx
                            )
                        ),
                        normalize=True,
                        nrow=(x_res_ema.size(0) // (x_src.size(0) + 2) + 1)
                    )