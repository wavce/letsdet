from utils.register import Register


BACKBONES = Register("backbones")
NECKS = Register("necks")
HEADS = Register("heads")
DETECTORS = Register("detectors")


def build_backbone(backbone, **kwargs):
    return BACKBONES[backbone](**kwargs)


def build_neck(neck, **kwargs):
    return NECKS[neck](**kwargs)


def build_head(head, **kwargs):
    return HEADS[head](**kwargs)


def build_detector(detector, **kwargs):
    return DETECTORS[detector](**kwargs)

