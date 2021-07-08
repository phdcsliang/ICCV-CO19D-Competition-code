class DefaultConfigs(object):
    #1.string parameters
    train_data = "./"
    test_data = "./"
    val_data = "no"
    model_name = "ctnet18"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0,1"

    #2.numeric parameters
    epochs = 120
    batch_size = 32
    img_height = 224
    img_weight = 224
    num_classes = 2
    CT_nums = 32
    seed = 666
    lr = 1e-2
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
