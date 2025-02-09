import typing

from torch.utils.data import ConcatDataset, DataLoader
from transformers import PreTrainedTokenizer

from invoke_training.training.config.data_config import DreamBoothDataLoaderConfig
from invoke_training.training.shared.data.data_loaders.dreambooth_sd_dataloader import (
    InterleavedSampler,
    SequentialRangeSampler,
    ShuffledRangeSampler,
)
from invoke_training.training.shared.data.data_loaders.image_caption_sdxl_dataloader import (
    sdxl_image_caption_collate_fn,
)
from invoke_training.training.shared.data.datasets.image_dir_dataset import (
    ImageDirDataset,
)
from invoke_training.training.shared.data.datasets.transform_dataset import (
    TransformDataset,
)
from invoke_training.training.shared.data.transforms.constant_field_transform import (
    ConstantFieldTransform,
)
from invoke_training.training.shared.data.transforms.drop_field_transform import (
    DropFieldTransform,
)
from invoke_training.training.shared.data.transforms.load_cache_transform import (
    LoadCacheTransform,
)
from invoke_training.training.shared.data.transforms.sd_tokenize_transform import (
    SDTokenizeTransform,
)
from invoke_training.training.shared.data.transforms.sdxl_image_transform import (
    SDXLImageTransform,
)
from invoke_training.training.shared.data.transforms.tensor_disk_cache import (
    TensorDiskCache,
)


def build_dreambooth_sdxl_dataloader(
    data_loader_config: DreamBoothDataLoaderConfig,
    tokenizer_1: PreTrainedTokenizer,
    tokenizer_2: PreTrainedTokenizer,
    batch_size: int,
    vae_output_cache_dir: typing.Optional[str] = None,
    shuffle: bool = True,
    sequential_batching: bool = False,
) -> DataLoader:
    """Construct a DataLoader for a DreamBooth dataset for Stable Diffusion XL.

    Args:
        data_loader_config (DreamBoothDataLoaderConfig):
        tokenizer_1 (PreTrainedTokenizer): Tokenizer 1.
        tokenizer_2 (PreTrainedTokenizer): Tokenizer 2.
        batch_size (int):
        vae_output_cache_dir (str, optional): The directory where VAE outputs are cached and should be loaded from. If
            set, then the image augmentation transforms will be skipped, and the image will not be copied to VRAM.
        shuffle (bool, optional): Whether to shuffle the dataset order.
        sequential_batching (bool, optional): If True, the internal dataset will be processed sequentially rather than
            interleaving class and instance examples. This is intended to be used when processing the entire dataset for
            caching purposes. Defaults to False.

    Returns:
        DataLoader
    """
    # 1. Prepare instance dataset
    instance_dataset = ImageDirDataset(data_loader_config.instance_data_dir, id_prefix="instance_")
    instance_dataset = TransformDataset(
        instance_dataset,
        [
            ConstantFieldTransform("caption", data_loader_config.instance_prompt),
            ConstantFieldTransform("loss_weight", 1.0),
        ],
    )
    datasets = [instance_dataset]

    # 2. Prepare class dataset.
    class_dataset = None
    if data_loader_config.class_data_dir is not None:
        class_dataset = ImageDirDataset(data_loader_config.class_data_dir, id_prefix="class_")
        class_dataset = TransformDataset(
            class_dataset,
            [
                ConstantFieldTransform("caption", data_loader_config.class_prompt),
                ConstantFieldTransform("loss_weight", data_loader_config.class_data_loss_weight),
            ],
        )
        datasets.append(class_dataset)

    # 3. Merge instance dataset and class dataset.
    merged_dataset = ConcatDataset(datasets)
    all_transforms = [
        SDTokenizeTransform(tokenizer_1, src_caption_key="caption", dst_token_key="caption_token_ids_1"),
        SDTokenizeTransform(tokenizer_2, src_caption_key="caption", dst_token_key="caption_token_ids_2"),
    ]
    if vae_output_cache_dir is None:
        all_transforms.append(
            SDXLImageTransform(
                resolution=data_loader_config.image_transforms.resolution,
                center_crop=data_loader_config.image_transforms.center_crop,
                random_flip=data_loader_config.image_transforms.random_flip,
            )
        )
    else:
        vae_cache = TensorDiskCache(vae_output_cache_dir)
        all_transforms.append(
            LoadCacheTransform(
                cache=vae_cache,
                cache_key_field="id",
                cache_field_to_output_field={
                    "vae_output": "vae_output",
                    "original_size_hw": "original_size_hw",
                    "crop_top_left_yx": "crop_top_left_yx",
                },
            )
        )
        # We drop the image to avoid having to either convert from PIL, or handle PIL batch collation.
        all_transforms.append(DropFieldTransform("image"))
    merged_dataset = TransformDataset(merged_dataset, all_transforms)

    # 4. If sequential_batching is enabled, return a basic data loader that iterates over examples sequentially (without
    #    interleaving class and instance examples). This is typically only used when preparing the data cache.
    if sequential_batching:
        return DataLoader(
            merged_dataset,
            collate_fn=sdxl_image_caption_collate_fn,
            batch_size=batch_size,
            num_workers=data_loader_config.dataloader_num_workers,
            shuffle=shuffle,
        )

    # 5. Prepare instance dataset sampler. Note that the instance_dataset comes first in the merged_dataset.
    samplers = []
    if shuffle:
        samplers.append(SequentialRangeSampler(0, len(instance_dataset)))
    else:
        samplers.append(ShuffledRangeSampler(0, len(instance_dataset)))

    # 6. Prepare class dataset sampler. Note that the class_dataset comes first in the merged_dataset.
    if class_dataset is not None:
        if shuffle:
            samplers.append(SequentialRangeSampler(len(instance_dataset), len(instance_dataset) + len(class_dataset)))
        else:
            samplers.append(ShuffledRangeSampler(len(instance_dataset), len(instance_dataset) + len(class_dataset)))

    # 7. Interleave instance and class samplers.
    interleaved_sampler = InterleavedSampler(samplers)

    return DataLoader(
        merged_dataset,
        sampler=interleaved_sampler,
        collate_fn=sdxl_image_caption_collate_fn,
        batch_size=batch_size,
        num_workers=data_loader_config.dataloader_num_workers,
    )
