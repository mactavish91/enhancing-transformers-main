# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,5,6'
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.plugins import DeepSpeedPlugin
from enhancing.utils.general import get_config_from_file, initialize_from_config, setup_callbacks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='wds_vitvq_base')
    # parser.add_argument('--local_rank', default=0)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-nn', '--num_nodes', type=int, default=5)
    parser.add_argument('-ng', '--num_gpus', type=int, default=8)
    parser.add_argument('-u', '--update_every', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--base_lr', type=float, default=1e-4)
    parser.add_argument('-a', '--use_amp', default=False, action='store_false')
    parser.add_argument('-b', '--batch_frequency', type=int, default=10)
    parser.add_argument('-m', '--max_images', type=int, default=4)
    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(args.seed)

    # Load configuration
    config = get_config_from_file(Path("configs")/(args.config+".yaml"))
    exp_config = OmegaConf.create({"name": args.config, "epochs": args.epochs, "update_every": args.update_every,
                                   "base_lr": args.base_lr, "use_amp": args.use_amp, "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images})

    # Build model
    model = initialize_from_config(config.model)
    model.learning_rate = exp_config.base_lr

    # Setup callbacks
    callbacks, logger = setup_callbacks(exp_config, config)

    # Build data modules
    data = initialize_from_config(config.dataset)
    data.prepare_data()

    # Build trainer
    trainer = pl.Trainer(
                         max_steps=500000,
                         precision=16 if exp_config.use_amp else 32,
                         callbacks=callbacks,
                         # plugins=[DeepSpeedPlugin(config=args.ds_config)],
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         benchmark=True,
                         log_every_n_steps=10,
                         strategy="ddp",
                         accumulate_grad_batches=exp_config.update_every,
                         logger=logger)

    # Train
    trainer.fit(model, data)
