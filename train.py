import pickle
import warnings
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="train_vae")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    if config.trainer.memory_snapshot:
        snapshots_dir = Path("./snapshots")
        snapshots_dir.mkdir(exist_ok=True)
        torch.cuda.memory._record_memory_history()
    accelerator = Accelerator(
        gradient_accumulation_steps=config.trainer.accumulation_steps,
        mixed_precision=config.trainer.amp,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_random_seed(config.trainer.seed)
    OmegaConf.register_new_resolver("divide", lambda x, y: x // y)
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    if accelerator.is_main_process:
        writer = instantiate(config.writer, logger, project_config)
    else:
        writer = None

    device = accelerator.device

    # if config.trainer.device == "auto":
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    # else:
    #     device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    if accelerator.is_main_process:
        logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    config.lr_scheduler.steps_per_epoch *= accelerator.num_processes
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        accelerator=accelerator,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()
    if config.trainer.memory_snapshot:
        with open(
            snapshots_dir / f"rank{accelerator.process_index}.pickle",
            "wb",
        ) as output:
            pickle.dump(torch.cuda.memory._snapshot(), output)


if __name__ == "__main__":
    main()
