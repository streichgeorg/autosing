# Adapted from whisperspeech.train_multi

import math
import os
from pathlib import Path

import wandb

import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from whisperspeech import testing

class TrainingTask(pl.LightningModule):
    def __init__(self, model, model_hparams=None):
        super().__init__()
        self.model = model
        self.model_hparams = model_hparams
        
    def on_fit_start(self):
        if getattr(self.model, "setup"):
            self.model.setup(self.device)
        if self.model_hparams["torch_compile"] and getattr(self.model, "optimize_training"):
            import torch._dynamo
            torch._dynamo.config.optimize_ddp = False
            # FIXME: define a batch of dummy tensors in the model

            testing.test_model(model, train_ds, bs=batch_size)
            model.optimize_training()
    
    def configure_optimizers(self):
        """ Initialize AdamW optimizer"""
        lr = self.model_hparams["lr0"]
        weight_decay = self.model_hparams["weight_decay"]
        
        all_params = set(model.parameters())
        customized_params = set()
        groups = []
        group_map = {}
        for name,m in model.named_modules():
            if hasattr(m, "no_weight_decay") or hasattr(m, "lr_scale"):
                customized_params |= set(m.parameters())
                m_wd = 0 if hasattr(m, "no_weight_decay") else weight_decay
                m_lr = lr * getattr(m, "lr_scale", 1)
                group = group_map.get((m_wd, m_lr), None)
                if not group:
                    group = {"params": [], "names": [], "weight_decay": m_wd, "lr": m_lr}
                    groups.append(group)
                    group_map[(m_wd, m_lr)] = group
                group["params"] += m.parameters()
                group["names"].append(name)
                
        other_params = all_params - customized_params
        
        param_groups = groups + [
            {"names": ["other"], "params": list(other_params), "weight_decay": weight_decay },
        ]

        optimizer = torch.optim.AdamW(lr=lr, betas=(0.9, 0.95), params=param_groups)
        
        # modified from https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-1501597319
        def num_steps_per_epoch() -> int:
            """Get number of steps"""
            # Accessing _data_source is flaky and might break
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            # math.ceil so always overestimate (underestimating throws exceptions)
            num_steps = math.ceil(dataset_size / self.trainer.accumulate_grad_batches)
            return num_steps
        
        warmup_steps = self.model_hparams["warmup_steps"]
        total_steps = self.model_hparams["epochs"] * num_steps_per_epoch()
        self.model_hparams["pct_start"] = min(0.3, warmup_steps / total_steps)

        print(f"{self.model_hparams['epochs']=} epochs x {num_steps_per_epoch()=} steps")

        if self.model_hparams["lr_schedule"] == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                pct_start=self.model_hparams["pct_start"],
                max_lr=[pg.get("lr", lr) for pg in param_groups],
                steps_per_epoch=num_steps_per_epoch(),
                epochs=int(self.model_hparams["epochs"]),
                final_div_factor=25
            )
        elif self.model_hparams["lr_schedule"] == "linear":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1e-3, 1, warmup_steps
            )
            train_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1, 1/25, total_steps - warmup_steps
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_steps]
            )
        elif self.model_hparams["lr_schedule"] == "wsd":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1e-3, 1, warmup_steps
            )
            train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [int(total_steps - warmup_steps - 0.1*total_steps)], 1/8,
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_steps]
            )
        elif self.model_hparams["lr_schedule"] == "const":
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1, 1, warmup_steps
            )
        else:
            raise Exception("Unknown learning rate schedule")

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, train_batch, batch_idx):
        with torch.cuda.amp.autocast(cache_enabled=False):
            train_logits, train_loss = self.model.forward(*train_batch)

        total_loss = 0
        if isinstance(train_loss, dict):
            for k, v in train_loss.items():
                self.log(f"train/{k}", v, sync_dist=True)
                total_loss += v
        else:
            self.log("train_loss", train_loss, sync_dist=True)
            total_loss = train_loss

        return total_loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        with torch.cuda.amp.autocast(cache_enabled=False):
            val_logits, val_loss = self.model.forward(*val_batch)

        total_loss = 0
        if isinstance(val_loss, dict):
            for k, v in val_loss.items():
                self.log(f"val/{k}", v, sync_dist=True)
                total_loss += v
        else:
            self.log("val_loss", val_loss, sync_dist=True)
            total_loss = val_loss

        return total_loss

    def on_validation_epoch_end(self):
        if hasattr(self.model, "get_metrics"):
            self.log_dict({"metrics/"+k:v for k,v in self.model.get_metrics().items()}, sync_dist=True)
    
    def test_step(self, val_batch, batch_idx):
        test_logits, test_loss = self.model.forward(*val_batch)

        self.log("test_loss", test_loss, sync_dist=True)
        return test_loss

def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

import json
import shlex

# watch out: we can only pass Python values as keyword arguments (not positional)
# everything else has to be a string
def parse_and_call(name, fun, args, kwargs={}, log_to_wandb=True):
    print(f"Parsing arguments for {name}, {args}")

    if len(args) > 0: args = json.loads(args)
    else: args = dict()

    args = {**args, **kwargs}

    if log_to_wandb and type(wandb_logger.experiment.config) is wandb.sdk.wandb_config.Config:
        wandb_logger.experiment.config[name] = {k:v for k,v in args.items() if k not in ["dataset", "tunables"]}
    return fun(**args)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, help="Task to train")
parser.add_argument("--task_args", type=str, help="Task to train")
parser.add_argument("--seed", type=int, default=0, help="Global training seed")
parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
parser.add_argument("--workers", type=int, default=4, help="max dataloader workers (per RANK in DDP mode)")
parser.add_argument("--input-dir", type=str, default="", help="input data path") # fixed in the model for now
parser.add_argument("--dataset-config", type=str, default="{}", help="common dataset options")
parser.add_argument("--monitored-metric", type=str, default="val_loss", help="metric to monitor for checkpointing")
parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/", help="directory to save the checkpoints")
parser.add_argument("--epochs", type=int, default=5, help="total training epochs")
parser.add_argument("--validate-every-n-steps", type=int, default=500, help="how training steps to run between validations")
parser.add_argument("--lr-schedule", type=str, default="cosine", help="the learning rate schedule [cosine, linear or wsd]")
parser.add_argument("--clip-gradient-norm", type=float, default=None, help="enable gradient norm clipping")
parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="perform the optimizer step only after going through several batches of samples")
parser.add_argument("--precision", type=str, default="16-mixed", help="floating point precision")
parser.add_argument("--torch-compile", type=bool, default=False, help="compile (parts of) the model with torch.compile")
parser.add_argument("--tunables", type=str, default="", help="tunable hyperparameters")
parser.add_argument("--resume-from", type=Path, default=None, help="resume training from the given checkpoint")
parser.add_argument("--load-from", type=Path, default=None, help="initialize the weights from the given model")
parser.add_argument("--strategy", type=str, default="ddp", help="distributed training strategy")
parser.add_argument("--wandb-suffix", type=str, default=None, help="W&B project name suffix")
parser.add_argument("--wandb-task-name", type=str, default=None, help="Task name for the W&B project name")
parser.add_argument("--shufbuf", type=int, default=64, help="Size of shuffel buffer (k)")
parser.add_argument("--output-dir", type=str, default="./models/")
parser.add_argument("--fancy-profiling", type=bool, default=False)
parser.add_argument("--persistent", type=bool, default=False)
parser.add_argument("--single-epoch", type=bool, default=False)

args = parser.parse_args().__dict__

task_name: list = args.pop("task_name")
task_args = args.pop("task_args")
input_args: list = shlex.split(args.pop("input_dir"))
dataset_config = json.loads(args.pop("dataset_config"))
monitored_metric: str = args.pop("monitored_metric")
checkpoint_dir: str = args.pop("checkpoint_dir")
num_workers: int = args.pop("workers")
batch_size: int = args.pop("batch_size")
epochs: int = args.pop("epochs")
tunables_args: str = args.pop("tunables")
shufbuf_size: int = args.pop("shufbuf")
output_dir: str = args.pop("output_dir")
Path(f"{output_dir}/{task_name}").mkdir(exist_ok=True, parents=True)
fancy_profiling: bool = args.pop("fancy_profiling")

assert not (fancy_profiling and args["torch_compile"])

hyp_params = {}
hyp_params["batch_size"] = batch_size
hyp_params["accumulate_grad_batches"] = args["accumulate_grad_batches"]
hyp_params["validate_every_n_steps"] = args["validate_every_n_steps"]
hyp_params["precision"] = args["precision"]
hyp_params["torch_compile"] = args["torch_compile"]
hyp_params["lr_schedule"] = args["lr_schedule"]
hyp_params["epochs"] = epochs
hyp_params["strategy"] = args["strategy"]

if "SLURM_NTASKS" in os.environ:
    hyp_params["world_size"] = os.environ["SLURM_NTASKS"]
else:
    hyp_params["world_size"] = 1

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.fabric.utilities.rank_zero import rank_zero_only
import webdataset as wds
import importlib
import dataclasses

torch.set_float32_matmul_precision("medium")

project = f"WhisperSpeech-{args['wandb_task_name'] or task_name}"
if args["wandb_suffix"]:
    project += "-"+args["wandb_suffix"]

from faker import Faker
fake = Faker()
name = (fake.name().split()[0] + "_" + fake.color_name()).lower()

if rank_zero_only.rank == 0:
    print("Experiment name:", name)
wandb_logger = WandbLogger(project=project, name=name)

class GradNormCallback(Callback):
    def on_after_backward(self, trainer, model):
        model.log("grad_norm", gradient_norm(model))


ckpt_callback = pl.callbacks.ModelCheckpoint(
     dirpath=f"{output_dir}/{task_name}",
     filename=f"{task_name}-{name}"+"-{epoch}",
     save_top_k=-1,
)

lr_monitor_callback = LearningRateMonitor(logging_interval="step")
grad_norm_callback = GradNormCallback()

task = importlib.import_module("autosing."+task_name)

# load all training sets
train_ds = task.load_dataset("train", **dataset_config)
train_total_samples = train_ds.total_samples
train_total_batches = int(train_total_samples / batch_size / int(hyp_params["world_size"]))
if train_total_batches < hyp_params["validate_every_n_steps"]:
    # validate once at the end of every epoch for very short experiments
    hyp_params["validate_every_n_steps"] = train_total_batches * 2

print(f"Training on {train_ds.total_length // 3600} hours")

val_ds = task.load_dataset("val", **dataset_config)

torch.multiprocessing.set_sharing_strategy("file_system")

train_loader = wds.WebLoader(
    train_ds,
    num_workers=num_workers,
    batch_size=None,
    shuffle=False, persistent_workers=args["persistent"],
).map(
    lambda t: tuple(x.clone() for x in t)
).unbatched().batched(batch_size)

train_loader = train_loader.repeat(
    nepochs=100,
    nbatches=train_total_batches,
).with_length(train_total_batches)

train_loader = train_loader.unbatched().shuffle(
    1024 * shufbuf_size,
    initial=1024 * shufbuf_size
).batched(batch_size)

val_loaders = [
    wds.WebLoader(
        val_ds, num_workers=0,
        drop_last=False, batch_size=None, shuffle=False,
    ).unbatched().batched(batch_size).with_length(val_ds.total_samples // batch_size)
]

assert hasattr(task, "Tunables")

tunables = parse_and_call("tunables", task.Tunables, tunables_args, log_to_wandb=False)
# override command line args from the tunables object
for k in ["lr0", "clip_gradient_norm", "weight_decay", "warmup_steps"]:
    val = getattr(tunables, k, None)
    if val is not None: hyp_params[k] = val

if type(wandb_logger.experiment.config) is wandb.sdk.wandb_config.Config:
    wandb_logger.experiment.config["tunables"] = dataclasses.asdict(tunables)

if args["load_from"]:
    model = task.load_model(str(args["load_from"]))
else:
    model_kwargs = dict(dataset=train_ds)
    if tunables is not None: model_kwargs["tunables"] = tunables
    model = parse_and_call("model", task.make_model, task_args, model_kwargs)
    
task = TrainingTask(model, model_hparams=hyp_params)

if fancy_profiling:
    profiler = pl.profilers.PyTorchProfiler(export_to_chrome=True)
else:
    profiler = "simple"

class AbortCallbacK(Callback):
    def __init__(self, start_epoch):
        self.num_epochs = 0

    def on_train_epoch_start(self, trainer, model):
        self.num_epochs += 1
        if self.num_epochs > 1:
            raise Exception("Abort training")

if args["single_epoch"]:
    abort_callback = [AbortCallbacK(task.current_epoch)]
else:
    abort_callback = []


trainer = pl.Trainer(strategy=hyp_params["strategy"],
                  max_epochs=hyp_params["epochs"],
                  accelerator="gpu",
                  profiler=profiler,
                  precision=hyp_params["precision"],
                  gradient_clip_val=hyp_params["clip_gradient_norm"],
                  accumulate_grad_batches=hyp_params["accumulate_grad_batches"],
                  # val_check_interval=hyp_params["validate_every_n_steps"],
                  check_val_every_n_epoch=1,
                  enable_checkpointing=True,
                  logger=wandb_logger,
                  num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
                  devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
                  callbacks=[ckpt_callback, lr_monitor_callback, *abort_callback])

# We can only get the rank after constructing the trainer
train_ds.ddp_rank = trainer.global_rank
train_ds.ddp_world_size = trainer.world_size
val_ds.ddp_rank = trainer.global_rank
val_ds.ddp_world_size = trainer.world_size

if type(wandb_logger.experiment.config) is wandb.sdk.wandb_config.Config:
    wandb_logger.experiment.config.update(hyp_params)

torch.multiprocessing.set_sharing_strategy("file_system")

kwargs = {}
if "resume_from" in args:
    kwargs["ckpt_path"] = args["resume_from"]
trainer.fit(model=task, train_dataloaders=train_loader, val_dataloaders=val_loaders, **kwargs)

if rank_zero_only.rank == 0:
    fname = f"{output_dir}/{task_name}/{name}.model"
    print("Saving:", fname)
    model.save_model(fname)
