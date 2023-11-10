import os

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo

from loss import ValidationLoss

import cv2


def get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes):
    
    cfg = _get_cfg()

    # Merge the model's default configuration file with the default Detectron2 configuration file.
    cfg.merge_from_file(model_zoo.get_config_file(model))

    # Set the training and validation datasets and exclude the test dataset.
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("val",)
    cfg.DATASETS.TEST = ()

    # Set the device to use for training.
    if device in ['cpu']:
        cfg.MODEL.DEVICE = 'cpu'

    # Set the number of data loader workers.
    cfg.DATALOADER.NUM_WORKERS = 2

    # Set the model weights to the ones pre-trained on the COCO dataset.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Set the batch size used by the solver.
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # Set the checkpoint period.
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    # Set the base learning rate.
    cfg.SOLVER.BASE_LR = learning_rate

    # Set the maximum number of training iterations.
    cfg.SOLVER.MAX_ITER = iterations

    # Set the learning rate scheduler steps to an empty list, which means the learning rate will not be decayed.
    cfg.SOLVER.STEPS = []

    # Set the batch size used by the ROI heads during training.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Set the number of classes.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes

    # Set the output directory.
    cfg.OUTPUT_DIR = output_dir

    return cfg


def get_dicts(img_dir, ann_dir):
    
    dataset_dicts = []
    for idx, file in enumerate(os.listdir(ann_dir)):
        # annotations should be provided in yolo format

        record = {}

        filename = os.path.join(img_dir, file[:-4] + '.jpg')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(os.path.join(ann_dir, file)) as r:
            lines = [l[:-1] for l in r.readlines()]

        for _, line in enumerate(lines):
            if len(line) > 2:
                label, cx, cy, w_, h_ = line.split(' ')

                obj = {
                    "bbox": [int((float(cx) - (float(w_) / 2)) * width),
                             int((float(cy) - (float(h_) / 2)) * height),
                             int(float(w_) * width),
                             int(float(h_) * height)],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": int(label),
                }

                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_datasets(root_dir, class_list_file):
    
    # Read the list of class names from the class list file.
    with open(class_list_file, 'r') as reader:
        classes_ = [l[:-1] for l in reader.readlines()]

    # Register the train and validation datasets.
    for d in ['train', 'val']:
        DatasetCatalog.register(d, lambda d=d: get_dicts(os.path.join(root_dir, d, 'imgs'),
                                                         os.path.join(root_dir, d, 'anns')))
        # Set the metadata for the dataset.
        MetadataCatalog.get(d).set(thing_classes=classes_)

    return len(classes_)


def train(output_dir, data_dir, class_list_file, learning_rate, batch_size, iterations, checkpoint_period, device,
          model):
    

    # Register the dataset and get the number of classes
    nmr_classes = register_datasets(data_dir, class_list_file)

    # Get the configuration for the model
    cfg = get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes)

    # Create the output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create the trainer object
    trainer = DefaultTrainer(cfg)

    # Create a custom validation loss object
    val_loss = ValidationLoss(cfg)

    # Register the custom validation loss object as a hook to the trainer
    trainer.register_hooks([val_loss])

    # Swap the positions of the evaluation and checkpointing hooks so that the validation loss is logged correctly
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    # Resume training from a checkpoint or load the initial model weights
    trainer.resume_or_load(resume=False)

    # Train the model
    trainer.train()
