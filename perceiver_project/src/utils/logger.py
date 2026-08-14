import os

# wandb e' importato pigramente (solo con --use_wandb), come gia' tensorboard
# qui sotto: cosi' non e' una dipendenza obbligatoria del progetto.


class BaseLogger:
    def __init__(self, log_dir='./logs', experiment_name='default_experiment', use_tensorboard=True, use_wandb=False, wandb_project_name=None):
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.writer = None

        if self.use_tensorboard:
            # Imported lazily: some local environments have a broken
            # tensorboard/protobuf/numpy combo, and this dependency is only
            # needed when --use_tensorboard is actually requested.
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)

        if self.use_wandb:
            import wandb
            if wandb_project_name is None:
                wandb_project_name = "perceiver-project"
            wandb.init(project=wandb_project_name, name=experiment_name, dir=log_dir)

    def log_scalar(self, tag, value, step):
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_hparams(self, hparams_dict, metrics_dict):
        if self.use_tensorboard and self.writer:
            self.writer.add_hparams(hparams_dict, metrics_dict)
        if self.use_wandb:
            import wandb
            wandb.config.update(hparams_dict)
            wandb.log(metrics_dict)

    def close(self):
        if self.use_tensorboard and self.writer:
            self.writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()
