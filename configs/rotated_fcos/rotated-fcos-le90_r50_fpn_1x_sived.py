_base_ = [
    '../_base_/datasets/sived.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# 模型配置
model = dict(
    type='mmdet.FCOS',
    init_cfg=dict(type='Pretrained', checkpoint='weights/FSModel/Rotated-FCOS.pth'),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        # frozen_stages=4, #todo 冻结所有骨干层
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True), #todo 冻结BN统计量
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=6, # 保持输出6个类别
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_angle=None,
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    
        train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)
)

# 优化器配置 (仅训练检测头)
# optim_wrapper=dict(
#     optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone':dict(lr_mult=0.0), # 冻结骨干网络
#             'neck':dict(lr_mult=0.5),     # 降低neck学习率
#             'bbox_head':dict(lr_mult=1.0) # 正常训练检测头
#         })
#     # clip_grad=dict(max_norm=35, norm_type=2))
# )

# 学习率调整策略
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1.0 / 3,
#         by_epoch=False,
#         begin=0,
#         end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]
