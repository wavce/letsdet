from utils.register import Register


DATASETS = Register("dataset")
AUGMENTATIONS = Register("augmentations")


def build_dataset(dataset, **kwargs):
    return DATASETS[dataset](**kwargs).dataset()


def build_augmentation(augmentation, **kwargs):
    return AUGMENTATIONS[augmentation](**kwargs)

