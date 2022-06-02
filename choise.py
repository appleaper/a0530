from util.logger import init_logger

def fanpai_train():
    from config.configV2 import ConfigV2
    from dataset.fanpai_datasetV2 import TrainFanpaiV2, ValFanpaiV2
    from torch.utils.data import DataLoader
    from trainer.trainV2 import TrainMainV2


    conf = ConfigV2()
    logging = init_logger(log_dir=conf.log_dir, stdout=True, name=conf.model_name)
    train_dataset = TrainFanpaiV2(conf)
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=4
    )

    val_dataset = ValFanpaiV2(conf)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(conf.batch_size * 1.5),
        shuffle=True,
        pin_memory=False,
        num_workers=4
    )

    logging.info(f'train num: {len(train_loader) * conf.batch_size}, val num: {len(val_loader) * conf.batch_size}')
    train_class2 = TrainMainV2(conf, train_loader, val_loader, logging)
    train_class2._train_stage()

def gener_train():
    from config.configV1 import ConfigV1
    from dataset.fanpai_datasetV1 import TrainFanpaiV1, ValFanpaiV1
    from torch.utils.data import DataLoader
    from trainer.trainV1 import TrainMainV1

    conf = ConfigV1()
    logging = init_logger(log_dir=conf.log_dir, stdout=True, name=conf.model_name)
    train_dataset = TrainFanpaiV1(conf)
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=4
    )

    val_dataset = ValFanpaiV1(conf)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(conf.batch_size * 1.5),
        shuffle=False,
        pin_memory=False,
        num_workers=4
    )

    logging.info(f'train num: {len(train_loader) * conf.batch_size}, val num: {len(val_loader) * conf.batch_size}')
    train_class1 = TrainMainV1(conf, train_loader, val_loader, logging)     # 初始化模型
    train_class1._train_stage()     # 模型训练
