from utils.register import Register


ASSIGNERS = Register(name="assigners")

SAMPLERS = Register(name="samplers")

LOSSES = Register(name="losses")

OPTIMIZERS = Register(name="optimizers")

LR_SCHEDULERS = Register(name="lr_schedulers")

METRICS = Register(name="metrics")

ANCHOR_GENERATORS = Register(name="anchor_generator")

NMS = Register(name="nms")


def build_assigner(assigner, **kwargs):
    return ASSIGNERS[assigner](**kwargs)


def build_sampler(sampler, **kwargs):
    return SAMPLERS[sampler](**kwargs)


def build_loss(loss, **kwargs):
    return LOSSES[loss](**kwargs)


def build_learning_rate_scheduler(scheduler, **kwargs):
    return LR_SCHEDULERS[scheduler](**kwargs)


def build_metric(metric, **kwargs):
    return METRICS[metric](**kwargs)


def build_optimizer(optimizer, **kwargs):
    return OPTIMIZERS[optimizer](**kwargs)


def build_nms(nms, **kwargs):
    return NMS[nms](**kwargs)


def build_anchor_generator(generator, **kwargs):
    return ANCHOR_GENERATORS[generator](**kwargs)
    