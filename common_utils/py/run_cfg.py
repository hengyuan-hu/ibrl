import os


class RunConfig:
    """Base config class to handle wandb, log and cfg dump"""

    use_wb: int = 0
    save_dir: str = ""

    @property
    def wb_exp(self):
        return None if not self.use_wb else self.save_dir.split("/")[-2]

    @property
    def wb_run(self):
        return None if not self.use_wb else self.save_dir.split("/")[-1]

    @property
    def wb_group(self):
        if not self.use_wb:
            return None
        else:
            return "_".join([w for w in self.wb_run.split("_") if "seed" not in w])  # type: ignore

    @property
    def cfg_path(self):
        return os.path.join(self.save_dir, "cfg.yaml")

    @property
    def log_path(self):
        return os.path.join(self.save_dir, "train.log")
