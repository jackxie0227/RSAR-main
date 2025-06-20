_base_ = [
    '../_base_/datasets/sived.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='H2RBoxV2Detector', # TO mmrotate/models/detectors/h2rbox_v2.py
    crop_size=(800, 800), # 中心尺寸裁切
    view_range=(0.25, 0.75), # 随机旋转角度范围 [π/4 ~ 3π/4]
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32, # 填充图像使尺寸为 32 倍数
        boxtype2tensor=False),
    backbone=dict( #* 主干网络
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), #TODO 加载ImageNet权重？？？
    neck=dict( #* 颈部框提取网络
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict( #* 检测头网络
        type='H2RBoxV2Head', # # TO mmrotate/models/dense_heads/h2rbox_v2_head.py
        num_classes=6,
        in_channels=256,
        angle_version='le90',
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=False,
        rotation_agnostic_classes=[3],
        agnostic_resize_classes=[],
        use_circumiou_loss=True, #! 数据集水平框时选择 False
        use_standalone_angle=True,
        use_reweighted_loss_bbox=False,
        angle_coder=dict(
            type='UCResolver', # mmrotate\models\task_modules\coders\angle_coder.py
            angle_version=angle_version,
            mdim=2,
            invalid_thr=0.2,
            loss_angle_restrict=dict(
                type='mmdet.L1Loss', loss_weight=0.05),
            ),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_symmetry_ss=dict(
            type='H2RBoxV2ConsistencyLoss', # TO mmrotate/models/losses/h2rbox_v2_consistency_loss.py
            use_snap_loss=True,
            loss_rot=dict(
                type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
            loss_flp=dict(
                type='mmdet.SmoothL1Loss', loss_weight=0.05, beta=0.1))),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# load hbox annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'), # 以四边形格式加载标注信息
    # Horizontal GTBox, (x1,y1,x2,y2)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')), # 转换为水平框（最小外界矩形）
    # Horizontal GTBox, (x,y,w,h,theta)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')), # 转换为旋转框 此时角度为 0°
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.Pad', size=(800, 800), pad_val=dict(img=(0, 0, 0))),  # 黑边填充至 (800, 800)
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']), # 水平翻转、垂直翻转、对角翻转
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=6)
