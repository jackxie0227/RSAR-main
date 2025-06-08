from argparse import ArgumentParser

import mmcv
from mmdet.apis import init_detector

from mmrotate.apis import inference_detector_by_patches
from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[512], 
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[500],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0], 
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr', 
        type=float,
        default=0.2,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--merge_nms_type',
        default='nms_rotated',
        choices=['nms', 'nms_rotated', 'nms_quadri'],
        help='NMS type for merging results')
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
    args = parser.parse_args()
    return args

import os
def is_image_file(path, extensions):
    """Check if the path is a valid image file."""
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

import time
def process_single_image(model, visualizer, img_path, out_dir, nms_cfg, patch_steps, patch_sizes, img_ratios):
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    
    start_time = time.time()
    result = inference_detector_by_patches(
        model, img, patch_sizes, patch_steps, img_ratios, nms_cfg)
    elapsed = time.time() - start_time
    
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(img_path)
    out_path = os.path.join(out_dir, f"{filename}")
    
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=out_path,
        pred_score_thr=args.score_thr)
    
    return elapsed, out_path

from datetime import datetime
def main(args):
    # register all modules in mmrotate into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # test a huge image by patches
    nms_cfg = dict(type=args.merge_nms_type, iou_threshold=args.merge_iou_thr)
    
    processed_files = []
    if is_image_file(args.img, args.extensions):
        elapsed, out_path = process_single_image(
            model, visualizer, args.img, os.path.dirname(args.out_file), nms_cfg,
            args.patch_steps, args.patch_sizes, args.img_ratios)
        processed_files = [args.img]
    elif os.path.isdir(args.img):
        img_list = []
        for file in os.scandir(args.img):
            img_list.append(file.path)
        
        if not img_list:
            raise ValueError(f'No image files found in {args.img}')
        
        total_elapsed = 0.0
        for idx, img_path in enumerate(img_list):
            elapsed, _ = process_single_image(
                model, visualizer, img_path, args.out_file, nms_cfg,
                args.patch_steps, args.patch_sizes, args.img_ratios)
            processed_files.append(img_path)
            total_elapsed += elapsed
            print(f"Processed {idx + 1}/{len(img_list)}: {os.path.basename(img_path)}")
        num_images = len(img_list)
        avg_elapsed = total_elapsed / num_images if num_images > 0 else 0
        
        log_file = os.path.join(os.path.dirname(args.out_file), 'inference_log.txt')
        log_content = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n "
            f"Config: {args.config}\n"
            f"Checkpoint: {args.checkpoint}\n"
            f"Test Images: {num_images}\n"
            f"Average time per image: {avg_elapsed:.2f} seconds.\n"
            f"patch_sizes: {args.patch_sizes}\n"
            f"patch_steps: {args.patch_steps}\n"
            f"img_ratios: {args.img_ratios}\n"
            f"merge_iou_thr: {args.merge_iou_thr}\n"
            f"score_thr: {args.score_thr}\n"
        )

        with open(log_file, 'a') as f:
            f.write(log_content)
    else:
        raise ValueError(f"Unsupported input type: {args.img}. "
                         "Please provide a valid image file or directory.")


if __name__ == '__main__':
    args = parse_args()
    
    args.img = 'demo/input/hugeImages'
    args.config = 'configs/roi_trans/roi-trans-le90_r50_fpn_1x_sived_onlycar.py'
    args.checkpoint = 'weights/SIVEDModel/ROI_Transformer.pth'
    args.out_file = 'demo/output/hugeImages/ROI_Transformer'
    
    main(args)
