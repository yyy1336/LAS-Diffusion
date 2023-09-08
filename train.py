import fire
import os
from network.model_trainer import DiffusionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from utils.utils import exists
from pytorch_lightning import loggers as pl_loggers
from utils.utils import ensure_directory, run, get_tensorboard_dir, find_best_epoch
from utils.shapenet_utils import snc_category_to_synth_id_all
from torch.utils.tensorboard import SummaryWriter
import torch


def train_from_folder(
    data_form: int = 0, #0 for original method for the LAS-Diffusion project，1 for data_set_0.2(microstructure_v02)
    sdf_folder: str = "/home/D/dataset/shapenet_sdf",
    sketch_folder: str = "/home/D/dataset/shapenet_edge_our_new",
    data_class: str = "chair",
    results_folder: str = './results',
    name: str = "debug",
    image_size: int = 64,
    base_channels: int = 32,
    optimizier: str = "adam",
    attention_resolutions: str = "4, 8",
    lr: float = 2e-4,
    batch_size: int = 4,
    with_attention: bool = True,
    num_heads: int = 4,
    dropout: float = 0.1,
    noise_schedule: str = "linear",
    kernel_size: float = 2.0,
    ema_rate: float = 0.999,
    save_last: bool = True,
    verbose: bool = False,
    training_epoch: int = 200,
    in_azure: bool = False,
    new: bool = True,
    continue_training: bool = False,
    debug: bool = False,
    use_sketch_condition: bool = True,
    use_text_condition: bool = False,
    seed: int = 777,
    save_every_epoch: int = 20,
    gradient_clip_val: float = 1.,
    feature_drop_out: float = 0.1,
    data_augmentation: bool = False,
    view_information_ratio: float = 2.0,
    vit_global: bool = False,
    vit_local: bool = True,
    split_dataset: bool = False,
    elevation_zero: bool = False,
    detail_view: bool = False,
):
    if not in_azure:
        debug = True
    else:
        debug = False

    data_classes = list(snc_category_to_synth_id_all.keys())
    data_classes.extend(["debug","class_2", "class_5", "class_13", "all"])
    assert data_class in data_classes

    results_folder = results_folder + "/" + name
    ensure_directory(results_folder)
    if continue_training:
        new = False

    if new:
        run(f"rm -rf {results_folder}/*")


    model_args = dict(
#        tb_writer=tb_writer, #yyy
        data_form=data_form,
        results_folder=results_folder,
        sdf_folder=sdf_folder,
        sketch_folder=sketch_folder,
        data_class=data_class,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        noise_schedule=noise_schedule,
        use_sketch_condition=use_sketch_condition,
        use_text_condition=use_text_condition,
        base_channels=base_channels,
        optimizier=optimizier,
        attention_resolutions=attention_resolutions,
        with_attention=with_attention,
        num_heads=num_heads,
        dropout=dropout,
        ema_rate=ema_rate,
        verbose=verbose,
        save_every_epoch=save_every_epoch,
        kernel_size=kernel_size,
        training_epoch=training_epoch,
        gradient_clip_val=gradient_clip_val,
        debug=debug,
        image_feature_drop_out=feature_drop_out,
        view_information_ratio=view_information_ratio,
        data_augmentation=data_augmentation,
        vit_global=vit_global,
        vit_local=vit_local,
        split_dataset=split_dataset,
        elevation_zero=elevation_zero,
        detail_view=detail_view
    )
    seed_everything(seed)

    model = DiffusionModel(**model_args)
   
  
    if in_azure:
        try:
            log_dir = get_tensorboard_dir()
        except Exception as e:
            log_dir = results_folder
    else:
        log_dir = results_folder

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version=None,
        name='logs',
        default_hp_metric=False
    )

#    print(log_dir)
#     tb_writer = SummaryWriter (log_dir=log_dir) #yyy
#    print(tb_writer)
#    print(1)

    # TODO: Visualize the network model as a graph
    # init_occupancy=torch.zeros((batch_size,1,image_size,image_size,image_size))
   # init_t=torch.ones((16))
   # init_t=torch.tensor([-0.9068, -1.8652, -4.0730, -6.2264,  0.7031,  2.1166, -1.3802, -1.7290,
   #     -8.9910,  2.0590, -1.0757, -1.8619, -3.2745,  1.3725, -3.6298, -8.7354],
   #    device='cuda:0')
   #  tb_writer.add_graph(model.model.denoise_fn,init_occupancy)  #yyy: 'model.model.denoise_fn' is the unet, this line of code makes init_occupancy pass forward() and add the unet model to tb_writer

    # yyy: Visualize the network model as a graph
    # TODO：Here will make an error when you change the batch size and continue training，The current solution is simply to comment out these two sentences when continue training
    init_occupancy = torch.zeros((batch_size, 1, image_size, image_size, image_size))
    tb_logger.experiment.add_graph(model.model.denoise_fn, init_occupancy)

    
    checkpoint_callback = ModelCheckpoint(
        monitor="current_epoch",
        dirpath=results_folder,
        filename="{epoch:02d}",
        save_top_k=10,
        save_last=save_last,
        every_n_epochs=save_every_epoch,
        mode="max",
    )

    last_epoch = find_best_epoch(results_folder)
    if os.path.exists(os.path.join(results_folder, "last.ckpt")):
        last_ckpt = "last.ckpt"
    else:
        if exists(last_epoch):
            last_ckpt = f"epoch={last_epoch:02d}.ckpt"
        else:
            last_ckpt = "last.ckpt"

    find_unused_parameters = False
    if in_azure:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy=DDPPlugin(
                              find_unused_parameters=find_unused_parameters),
                          logger=tb_logger,
                          max_epochs=training_epoch,
                          log_every_n_steps=10,
                          callbacks=[checkpoint_callback])
    else:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy=DDPPlugin(
                              find_unused_parameters=find_unused_parameters),
                          logger=tb_logger,
                          max_epochs=training_epoch,
                          log_every_n_steps=1,
                          callbacks=[checkpoint_callback])

    if continue_training and os.path.exists(os.path.join(results_folder, last_ckpt)):
        trainer.fit(model, ckpt_path=os.path.join(results_folder, last_ckpt))
    else:
        trainer.fit(model)





if __name__ == '__main__':
    fire.Fire(train_from_folder)
