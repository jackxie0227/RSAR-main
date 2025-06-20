# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=100), #? warmup ori-500
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[12, 16], #? ori-[8, 11]
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD',
                   lr=0.0025,  #? ori-0.005
                   momentum=0.9,
                   weight_decay=0.0005), #? ori-0.0001
    clip_grad=dict(max_norm=35, norm_type=2))
