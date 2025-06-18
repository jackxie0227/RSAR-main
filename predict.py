import os
from argparse import Namespace
import mmcv
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules
from demo.draw_rotatedRect import draw_rotated_rectangles

def predict(patch_size: int, patch_step: int) -> str:
    """执行目标检测预测
    Args:
        patch_size: 检测块大小
        patch_step: 检测块步长
        
    Returns:
        str: 预测结果图片路径
    """
    try:
        # 注册所有模块
        register_all_modules()
        
        # 硬编码配置路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'configs/roi_trans/roi-trans-le90_r50_fpn_1x_sived.py')
        checkpoint_path = os.path.join(base_dir, 'weights/FSModel/ROI-Transformer-Adjust.pth')
        
        output_path = os.path.join(base_dir, 'apis/output/predimg.png')
        img_path = os.path.join(base_dir, 'apis/input/srcimg.png')
        
        # 创建模拟参数对象
        args = Namespace(
            config=config_path,
            checkpoint=checkpoint_path,
            out_file=output_path,
            patch_sizes=[patch_size],
            patch_steps=[patch_step],
            img_ratios=[1.0],
            merge_iou_thr=0.1,
            merge_nms_type='nms_rotated',
            device='cuda:0',
            palette='dota',
            score_thr=0.3
        )
        
        # 初始化模型
        model = init_detector(
            args.config, args.checkpoint, palette=args.palette, device=args.device)
        
        # 初始化可视化工具
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        
        # 分块推理
        nms_cfg = dict(type=args.merge_nms_type, iou_threshold=args.merge_iou_thr)
        result = inference_detector_by_patches(model, img_path, args.patch_sizes,
                                            args.patch_steps, args.img_ratios,
                                            nms_cfg)
        
        # 可视化并保存结果
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        save_img, _, _, _ = draw_rotated_rectangles(img, model.dataset_meta, result.pred_instances, args.score_thr)
            
        return output_path
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None

if __name__ == "__main__":
    predict(patch_size=512, patch_step=500)