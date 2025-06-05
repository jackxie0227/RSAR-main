from argparse import ArgumentParser
import torch
import mmcv
from mmdet.apis import inference_detector, init_detector
from datetime import datetime
from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image path or folder path')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help='Supported image extensions')
    parser.add_argument(
        '--logpath',
        default=None,
        help='Path to save log file')
    args = parser.parse_args()
    return args

import os
# 是否支持图像类型
def is_image_file(path, extensions): 
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

import time 
# 推理单张图像
def process_single_image(model, visualizer, img_path, out_dir, score_thr):
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    
    start_time = time.time()
    result = inference_detector(model, img)
    elapsed = time.time() - start_time
    # print('infer_time: ', elapsed)
    
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(img_path)
    out_path = os.path.join(out_dir, f"result_{filename}")
    
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=args.out_file is None,
        wait_time=0,
        out_file=out_path,
        pred_score_thr=score_thr)
    
    return elapsed, out_path

def main(args):
    register_all_modules() # 注册网络所有模块

    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)

    visualizer.dataset_meta = model.dataset_meta
    
    # 预热GPU
    # dummy_input = torch.randn(1, 3, 224, 224).to(args.device)
    # with torch.no_grad():
    #     model.forward(dummy_input)
    # torch.cuda.synchronize()  # 等待CUDA操作完成
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1e6  
    
    processed_files = []
    
    
    if is_image_file(args.img, args.extensions): # img 路径为图片路径
        elapsed, out_path = process_single_image(
            model, visualizer, args.img, args.out_dir, args.score_thr)
        processed_files = [args.img]
        print(f"Processed single image: {os.path.basename(args.img)}")
        print(f"Result saved to: {out_path}")
    elif os.path.isdir(args.img): # img 路径为文件夹路径
        img_list = []
        for file in os.scandir(args.img):
            img_list.append(os.path.join(file))
            
        if not img_list:
            raise FileNotFoundError(f"No images found in {args.img}")
        
        total_elapsed = 0.0
        for idx, img_path in enumerate(img_list):
            elapsed, out_path = process_single_image(
                model, visualizer, img_path, args.out_file, args.score_thr)
            if idx >= 1:
                total_elapsed += elapsed
            processed_files.append(img_path)
            print(f"[{idx+1}/{len(img_list)}] Processed: {os.path.basename(img_path)} costed {elapsed}")
        num_images = len(processed_files)
        avg_elapsed_time = total_elapsed / (num_images - 1) if num_images > 0 else 0
        
        # 记录 demo 测试信息
        log_file = os.path.join(args.logpath, "inference_log.txt")
        log_content = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
            f"Config: {args.config}\n"
            f"Checkpoint: {args.checkpoint}\n"
            f"Model Parameters: {total_params_m:.2f}M\n"
            f"Test Images: {num_images}\n"
            f"Avg Inference Time: {avg_elapsed_time:.4f}s\n"
            "----------------------------------------\n")
        with open(log_file, "a") as f:
            f.write(log_content)
    else:
        raise ValueError(f"Invalid input path: {args.img}. Must be a valid image file or directory.")



if __name__ == '__main__':
    args = parse_args()
    
    #! args.config 只能调用 configs 路径中的配置文件
    # 弱监督 Demo
    # args.img = 'demo/input/images'
    # args.config = 'configs/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_rsar_ucr_2d.py'
    # args.checkpoint = 'weights/WSModel/h2rbox_v2-le90_r50_fpn-1x_rsar_ucr_d2_epoch_12.pth'
    # args.out_file = 'demo/output/WSpred'
    
    # 全监督 Demo
    args.img = 'demo/input/images'
    args.config = 'configs/rotated_fcos/rotated-fcos-le90_r50_fpn_1x_rsar.py'
    args.checkpoint = 'weights/FSModel/Rotated-FCOS.pth'
    args.out_file = 'demo/output/FSpred'
    
    args.logpath = 'demo/output'
    
    main(args)
