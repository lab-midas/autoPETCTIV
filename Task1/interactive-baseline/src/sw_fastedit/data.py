from __future__ import annotations

import glob
import logging
import os
import shutil

# from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from monai.apps import CrossValidation
from monai.data import DataLoader, Dataset, ThreadDataLoader, partition_dataset

from monai.data.dataset import PersistentDataset
from monai.data.folder_layout import FolderLayout
from monai.transforms import (  # RandShiftIntensityd,; Resized,; ScaleIntensityRanged,; SignalFillEmpty
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    CopyItemsd,
    CropForegroundd,
    # DataStatsd,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Identityd,
    Invertd,
    Lambdad,
    LoadImaged,
    MeanEnsembled,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    SaveImaged,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SignalFillEmptyd,
    Spacingd,
    ToDeviced,
    # ToNumpyd,
    ToTensord,
    VoteEnsembled,
)
from monai.utils.enums import CommonKeys

from sw_fastedit.helper_transforms import (  # SignalFillEmptyd,
    # AbortifNaNd,
    CheckTheAmountOfInformationLossByCropd,
    InitLoggerd,
    PrintDatad,
    TrackTimed,
    threshold_foreground,
    cast_labels_to_zero_and_one,
)
from sw_fastedit.transforms import (
    AddEmptySignalChannels,
    AddGuidance,
    AddGuidanceJSON,
    AddGuidanceSignal,
    FindDiscrepancyRegions,
    NormalizeLabelsInDatasetd,
    SplitPredsLabeld,
    OverrideLabelsKeyd,
)
from sw_fastedit.utils.helper import convert_mha_to_nii, convert_nii_to_mha

logger = logging.getLogger("sw_fastedit")


PET_dataset_names = ["AutoPET", "AutoPET2", "AutoPET_merged", "HECKTOR", "AutoPET2_Challenge", "AutoPET4"]


def get_pre_transforms(labels: Dict, device, args, input_keys=("image", "label")):
    return Compose(get_pre_transforms_train_as_list(labels, device, args, input_keys)), Compose(
        get_pre_transforms_val_as_list(labels, device, args, input_keys)
    )


def get_spacing(args):
    AUTOPET_SPACING = (2.03642011, 2.03642011, 3.0)
    MSD_SPLEEN_SPACING = (2 * 0.79296899, 2 * 0.79296899, 5.0)
    # Apply this only to the label!
    #HECKTOR_SPACING = (4, 4, 4)
    HECKTOR_SPACING = (2.03642011, 2.03642011, 3.0)
    #HECKTOR_SPACING = (2,2,2)
    #HECKTOR_SPACING = (4 * 0.98, 4 * 0.98, 1 * 3.27)

    if args.dataset == "AutoPET" or args.dataset == "AutoPET4" or args.dataset == "AutoPET2" or args.dataset == "AutoPET2_Challenge":
        return AUTOPET_SPACING
    elif args.dataset == "HECKTOR":
        return HECKTOR_SPACING
    elif args.dataset == "MSD_Spleen":
        return MSD_SPLEEN_SPACING
    # return spacing


def get_pre_transforms_train_as_list(labels: Dict, device, args, input_keys=("image", "label")):
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # data Input keys have to be ["image", "label"] for train, and least ["image"] for val
    if args.dataset in PET_dataset_names:
        t = [
            # Initial transforms on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(
                keys=input_keys,
                reader="ITKReader",
                image_only=False,
                simple_keys=True,
            ),
            ToTensord(keys=input_keys, device=cpu_device, track_meta=True),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys="label", labels=labels, device=cpu_device, allow_missing_keys=True),
            Orientationd(keys=input_keys, axcodes="RAS"),
            # Spacingd(keys=input_keys, pixdim=spacing),
            Spacingd(keys='image', pixdim=spacing) if True else Identityd(keys=input_keys, allow_missing_keys=True),
            Spacingd(keys='label', pixdim=spacing, mode="nearest") if ('label' in input_keys) else Identityd(keys=input_keys, allow_missing_keys=True),
            # PrintDatad(),
            CropForegroundd(
                keys=input_keys,
                source_key="image",
                select_fn=threshold_foreground,
            )
            if args.crop_foreground
            else Identityd(keys=input_keys, allow_missing_keys=True),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically
            ScaleIntensityRanged(keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True)
            if args.use_scale_intensity_ranged
            else ScaleIntensityRangePercentilesd(
                keys="image", lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            # Random Transforms
            # allow_smaller=True not necessary for the default AUTOPET split of (224,)**3, just there for safety so that training does not get interrupted
            RandCropByPosNegLabeld(
                keys=input_keys,
                label_key="label",
                spatial_size=args.train_crop_size,
                pos=args.positive_crop_rate,
                neg=1 - args.positive_crop_rate,
                allow_smaller=True,
            )
            if args.train_crop_size is not None
            else Identityd(keys=input_keys, allow_missing_keys=True),
            DivisiblePadd(keys=input_keys, k=32, value=0)
            if args.inferer == "SimpleInferer"
            else Identityd(keys=input_keys, allow_missing_keys=True),  # UNet needs this, 32 for 6 layers, for 7 at least 64
            RandFlipd(keys=input_keys, spatial_axis=[0], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[1], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=input_keys, prob=0.10, max_k=3),
            # AbortifNaNd(input_keys),
            SignalFillEmptyd(input_keys),
            AddEmptySignalChannels(keys=input_keys, device=cpu_device)
            if not args.non_interactive
            else Identityd(keys=input_keys, allow_missing_keys=True),
            # PrintDatad(),
            # Move to GPU
            # WARNING: Activating the line below leads to minimal gains in performance
            # However you are buying these gains with a lot of weird errors and problems
            # So my recommendation after months of fiddling is to leave this off
            # Until MONAI has fixed the underlying issues
            # ToTensord(keys=("image", "label"), device=device, track_meta=False),
        ]

        if args.debug:
            for i in range(len(t)):
                t[i] = TrackTimed(t[i])
            print(t)

    return t


def get_pre_transforms_val_as_list(labels: Dict, device, args, input_keys=("image", "label")):
    cpu_device = torch.device("cpu")
    spacing = get_spacing(args)

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    input_keys = ("image") if args.docker else input_keys

    # data Input keys have to be at least ["image"] for val
    if args.dataset in PET_dataset_names:
        t = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys="label", labels=labels, device=cpu_device, allow_missing_keys=True) if ('label' in input_keys) else OverrideLabelsKeyd(keys=input_keys, allow_missing_keys=True, labels={"tumor": 1, "background": 0}),
            # Only for HECKTOR, filter out the values > 1
            Lambdad(keys="label", func=cast_labels_to_zero_and_one) if (args.dataset == "HECKTOR") else Identityd(keys=input_keys, allow_missing_keys=True),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys='image', pixdim=spacing) if True else Identityd(keys=input_keys, allow_missing_keys=True),
            Spacingd(keys='label', pixdim=spacing, mode="nearest") if ('label' in input_keys) else Identityd(keys=input_keys, allow_missing_keys=True),
            #PrintDatad(),
            CheckTheAmountOfInformationLossByCropd(
                keys="label", roi_size=args.val_crop_size, crop_foreground=args.crop_foreground
            )
            if "label" in input_keys
            else Identityd(keys=input_keys, allow_missing_keys=True),
            CropForegroundd(
                keys=input_keys,
                source_key="image",
                select_fn=threshold_foreground,
            )
            if args.crop_foreground
            else Identityd(keys=input_keys, allow_missing_keys=True),
            #PrintDatad(),
            CenterSpatialCropd(keys=input_keys, roi_size=args.val_crop_size)
            if args.val_crop_size is not None
            else Identityd(keys=input_keys, allow_missing_keys=True),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically
            ScaleIntensityRanged(keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True)
            if args.use_scale_intensity_ranged
            else ScaleIntensityRangePercentilesd(
                keys="image", lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            DivisiblePadd(keys=input_keys, k=32, value=0)
            if args.inferer == "SimpleInferer"
            else Identityd(keys=input_keys, allow_missing_keys=True),
            AddEmptySignalChannels(keys=input_keys, device=cpu_device)
            if not args.non_interactive
            else Identityd(keys=input_keys, allow_missing_keys=True),


        ]

    if args.debug:
        for i in range(len(t)):
            t[i] = TrackTimed(t[i])

    return t


def get_device(data):
    return f"device - {data.device}"


def get_click_transforms(device, args):
    spacing = get_spacing(args)
    cpu_device = torch.device("cpu")

    logger.info(f"{device=}")

    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        FindDiscrepancyRegions(keys="label", pred_key="pred", discrepancy_key="discrepancy", device=device),
        AddGuidance(
            keys="NA",
            discrepancy_key="discrepancy",
            probability_key="probability",
            device=device,
        ),
        # Overwrites the image entry
        AddGuidanceSignal(
            keys="image",
            sigma=args.sigma,
            disks=(not args.no_disks),
            device=device,
            # spacing=spacing,
        ),
        ToTensord(keys=("image", "label", "pred"), device=cpu_device)
        if args.sw_cpu_output
        else Identityd(keys=("pred",), allow_missing_keys=True),
    ]

    # if args.debug:
    #     for i in range(len(t)):
    #         t[i] = TrackTimed(t[i])

    return Compose(t)

def get_click_transforms_json(device, args, n_clicks=10):
    spacing = get_spacing(args)
    cpu_device = torch.device("cpu")

    logger.info(f"{device=}")

    t = [
        AddGuidanceJSON(
            keys="NA",
            json_dir=args.json_dir,
            n_clicks=n_clicks,
        ),
        # Overwrites the image entry
        AddGuidanceSignal(
            keys="image",
            sigma=args.sigma,
            disks=(not args.no_disks),
            device=device,
            # spacing=spacing,
        ),
        ToTensord(keys=("image", "label", "pred"), device=cpu_device)
        if args.sw_cpu_output
        else Identityd(keys=("pred",), allow_missing_keys=True),
    ]

    # if args.debug:
    #     for i in range(len(t)):
    #         t[i] = TrackTimed(t[i])

    return Compose(t)


def get_post_transforms(labels, *, save_pred=False, output_dir=None, pretransform=None, docker=False):
    cpu_device = torch.device("cpu")
    if save_pred:
        if output_dir is None:
            raise UserWarning("output_dir may not be empty when save_pred is enabled...")
        if pretransform is None:
            logger.warning("Make sure to add a pretransform here if you want the prediction to be inverted")

    input_keys = ("pred",)
    discrete_keys = input_keys if docker else ("pred", "label")
    argmax_vals = (True) if docker else (True, False)
    one_hot_vals = (len(labels)) if docker else (len(labels), len(labels))
    tensor_keys = ("image", "pred") if docker else ("image", "label", "pred")
    t = [
        CopyItemsd(keys=("pred",), times=1, names=("pred_for_save",))
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),
        Invertd(
            keys=("pred_for_save",),
            orig_keys="image",
            nearest_interp=False,
            transform=pretransform,
        )
        if (save_pred and pretransform is not None)
        else Identityd(keys=input_keys, allow_missing_keys=True),
        Activationsd(keys=("pred",), softmax=True),
        AsDiscreted(
            keys="pred_for_save",
            argmax=True,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),
        AsDiscreted(
            keys=discrete_keys,
            argmax=argmax_vals,
            to_onehot=one_hot_vals,
        ), 
        SaveImaged(
            keys=("pred_for_save",),
            writer="ITKWriter",
            output_dir=output_dir,
            output_postfix="",
            #    output_ext=".nii.gz",
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        )
        if save_pred
        else Identityd(keys=input_keys, allow_missing_keys=True),
        ToTensord(keys=tensor_keys, device=cpu_device),
        # This transform is to check dice score per segment/label
        # SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)


def get_post_transforms_unsupervised(labels, device, pred_dir, pretransform):
    os.makedirs(pred_dir, exist_ok=True)
    nii_layout = FolderLayout(output_dir=pred_dir, postfix="", extension=".nii.gz", makedirs=False)

    t = [
        Invertd(
            keys="pred",
            orig_keys="image",
            nearest_interp=False,
            transform=pretransform,
        ),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys="pred",
            argmax=True,  # to_onehot=(len(labels),),
        ),
        # This transform is to check dice score per segment/label, disabled not needed right now
        # SplitPredsLabeld(keys="pred"),
        SaveImaged(
            keys="pred",
            writer="ITKWriter",
            output_postfix="",
            output_ext=".nii.gz",
            folder_layout=nii_layout,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        ),
    ]
    return Compose(t)


def get_post_ensemble_transforms(labels, device, pred_dir, pretransform, nfolds=5, weights=None):
    prediction_keys = [f"pred_{i}" for i in range(nfolds)]

    os.makedirs(pred_dir, exist_ok=True)
    nii_layout = FolderLayout(output_dir=pred_dir, postfix="", extension=".nii.gz", makedirs=False)

    t = [
        Invertd(
            keys=prediction_keys,
            orig_keys="image",
            nearest_interp=False,
            transform=pretransform,
        ),
    ]

    mean_or_vote = "vote"
    if mean_or_vote == "mean":
        t += [
            EnsureTyped(keys=prediction_keys),
            MeanEnsembled(
                keys=prediction_keys,
                output_key="pred",
            ),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ]
    else:
        t += [
            EnsureTyped(keys=prediction_keys),
            Activationsd(keys=prediction_keys, softmax=True),
            AsDiscreted(keys=prediction_keys, argmax=True),
            VoteEnsembled(keys=prediction_keys, output_key="pred"),
        ]
    t += [
        SaveImaged(
            keys="pred",
            writer="ITKWriter",
            output_postfix="",
            output_ext=".nii.gz",
            folder_layout=nii_layout,
            output_dtype=np.uint8,
            separate_folder=False,
            resample=False,
        ),
    ]
    return Compose(t)


def get_val_post_transforms(labels, device):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)


def get_AutoPET_file_list(args) -> List[List, List, List]:
    train_images = sorted(glob.glob(os.path.join(args.input_dir, "imagesTr", "*0001.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.input_dir, "labelsTr", "*.nii.gz")))

    test_images = sorted(glob.glob(os.path.join(args.input_dir, "imagesTs", "*0001.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(args.input_dir, "labelsTs", "*.nii.gz")))

    train_data = [
        {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
    ]
    val_data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
    test_data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]

    return train_data, val_data, test_data


def get_filename_without_extensions(nifti_path):
    # Strips up to two extensions from the filename, e.g. SUV.nii.gz -> SUV
    return Path(os.path.basename(nifti_path)).with_suffix("").with_suffix("").name


def get_AutoPET2_Challenge_file_list(args) -> List[List, List, List]:
    test_images = sorted(glob.glob(os.path.join(args.input_dir, "*.mha")))

    logger.info(f"{test_images=}")
    test_data = []
    for image_path in test_images:
        logger.info(f"Converting {image_path} to .nii.gz")
        uuid = get_filename_without_extensions(image_path)
        nii_path = os.path.join(args.cache_dir, f"{uuid}.nii.gz")
        convert_mha_to_nii(image_path, nii_path)
        test_data.append({"image": nii_path})

    logger.info(f"{test_data=}")
    return [], [], test_data

def get_AutoPET4_Challenge_file_list(args) -> List[List, List, List]:
    test_images = sorted(glob.glob(os.path.join(args.input_dir, "*0001.nii.gz")))

    logger.info(f"{test_images=}")
    test_data = []
    for image_path in test_images:
        test_data.append({"image": image_path})

    logger.info(f"{test_data=}")
    return [], test_data, test_data

def get_AutoPET_file_list_without_docker(args) -> List[List, List, List]:

    test_images = sorted(glob.glob(os.path.join(args.input_dir, "images", "*0001.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(args.input_dir, "labels", "*.nii.gz")))
    
    val_data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
    test_data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]

    return [], val_data, test_data


def post_process_AutoPET2_Challenge_file_list(args, pred_dir, cache_dir):
    logger.info("POSTPROCESSING AutoPET challenge files")
    nii_dir = os.path.join(cache_dir, "prediction")
    shutil.move(pred_dir, nii_dir)
    os.makedirs(pred_dir, exist_ok=True)
    nii_images = sorted(glob.glob(os.path.join(nii_dir, "*.nii.gz")))
    logger.info(nii_images)

    for image_path in nii_images:
        logger.info(f"Using nii file {image_path}")
        image_name = get_filename_without_extensions(image_path)
        uuid = image_name
        logger.info(f"{uuid=}")

        mha_path = os.path.join(args.output_dir, f"{uuid}.mha")
        logger.info(f"Creating mha file {mha_path}")
        convert_nii_to_mha(image_path, mha_path)
        assert os.path.exists(mha_path)


def get_MSD_Spleen_file_list(args) -> List[List, List, List]:
    all_images = sorted(glob.glob(os.path.join(args.input_dir, "imagesTr", "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(args.input_dir, "labelsTr", "*.nii.gz")))

    data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]

    train_data, val_data = partition_dataset(
        data,
        ratios=[args.split, (1 - args.split)],
        shuffle=True,
        seed=args.seed,
    )
    return train_data, val_data, []


def get_AutoPET2_file_list(args) -> List[List, List, List]:
    all_images = []
    all_labels = []

    for root, _, files in os.walk(args.input_dir, followlinks=True):
        for file in files:
            if file.startswith("SUV") and file.endswith(".nii.gz"):
                all_images.append(os.path.join(root, file))
            if file.startswith("SEG") and file.endswith(".nii.gz"):
                all_labels.append(os.path.join(root, file))

    data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]

    train_data, val_data = partition_dataset(
        data,
        ratios=[args.split, (1 - args.split)],
        shuffle=True,
        seed=args.seed,
    )

    return train_data, val_data, []


def get_HECKTOR_file_list(args) -> List[List, List, List]:
    logger.warning("Run the resample2PET.py script before using HECKTOR. Different files have different spacings and I found no other way to deal with it..")

    # Assuming this is the folder /lsdf/data/medical/HECKTOR/hecktor2022_training/
    train_images = sorted(glob.glob(os.path.join(args.input_dir, "hecktor2022_training", "resampled", "*PT*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.input_dir, "hecktor2022_training", "resampled/labelsTr", "*.nii.gz")))

    test_images = sorted(glob.glob(os.path.join(args.input_dir, "hecktor2022_testing", "imagesTs", "*.nii.gz")))

    data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    logger.info(f"{data[-5:]=}")

    train_data, val_data = partition_dataset(
        data,
        ratios=[args.split, (1 - args.split)],
        shuffle=True,
        seed=args.seed,
    )

    test_data = [{"image": image_name} for image_name in test_images]
    return train_data, val_data, test_data


def get_data(args):
    logger.info(f"{args.dataset=}")

    test_data = []
    if args.docker is not None and args.docker:
        train_data, val_data, test_data = get_AutoPET4_Challenge_file_list(args)
    elif args.dataset == "AutoPET4":
        train_data, val_data, test_data = get_AutoPET_file_list_without_docker(args)
    elif args.dataset == "AutoPET":
        train_data, val_data, test_data = get_AutoPET_file_list(args)
    elif args.dataset == "AutoPET2_Challenge":
        train_data, val_data, test_data = get_AutoPET2_Challenge_file_list(args)
        return train_data, val_data, test_data
    elif args.dataset == "MSD_Spleen":
        train_data, val_data, test_data = get_MSD_Spleen_file_list(args)
    elif args.dataset == "AutoPET2":
        train_data, val_data, test_data = get_AutoPET2_file_list(args)
    elif args.dataset == "HECKTOR":
        train_data, val_data, test_data = get_HECKTOR_file_list(args)

    if args.train_on_all_samples:
        train_data += val_data
        val_data = train_data
        test_data = train_data
        logger.warning("All validation data has been added to the training. Validation on them no longer makes sense.")

    logger.info(f"len(train_data): {len(train_data)}, len(val_data): {len(val_data)}, len(test_data): {len(test_data)}")

    # For debugging with small dataset size
    train_data = train_data[0 : args.limit] if args.limit else train_data
    val_data = val_data[0 : args.limit] if args.limit else val_data
    test_data = test_data[0 : args.limit] if args.limit else test_data

    return train_data, val_data, test_data


def get_test_loader(args, pre_transforms_test):
    train_data, val_data, test_data = get_data(args)
    if not len(test_data):
        if len(val_data) > 0:
            test_data = val_data
        if len(train_data) > 0:
            test_data = train_data
        else:
            raise UserWarning("No valid data found..")

    total_l = len(test_data)
    test_ds = Dataset(test_data, pre_transforms_test)
    test_loader = DataLoader(
        test_ds,
        # shuffle=True,
        # num_workers=args.num_workers,
        batch_size=1,
        # The two options below are needed if ToDeviced('cuda' ,..) is activated..
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info("{} :: Total Records used for Testing is: {}".format(args.gpu, total_l))

    return test_loader


def get_train_loader(args, pre_transforms_train):
    train_data, val_data, test_data = get_data(args)
    total_l = len(train_data) + len(val_data)

    train_ds = PersistentDataset(train_data, pre_transforms_train, cache_dir=args.cache_dir)
    train_loader = ThreadDataLoader(
        train_ds,
        shuffle=True,
        num_workers=args.num_workers,
        batch_size=1,
        # The two options below are needed if ToDeviced('cuda' ,..) is activated..
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info("{} :: Total Records used for Training is: {}/{}".format(args.gpu, len(train_ds), total_l))

    return train_loader


def get_val_loader(args, pre_transforms_val):
    train_data, val_data, test_data = get_data(args)

    total_l = len(train_data) + len(val_data)

    val_ds = PersistentDataset(val_data, pre_transforms_val, cache_dir=args.cache_dir)
    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=args.num_workers,
        batch_size=1,
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info("{} :: Total Records used for Validation is: {}/{}".format(args.gpu, len(val_ds), total_l))

    return val_loader


def get_cross_validation(args, nfolds, pre_transforms_train, pre_transforms_val):
    folds = list(range(nfolds))

    train_data, val_data, test_data = get_data(args)

    cvdataset = CrossValidation(
        dataset_cls=PersistentDataset,
        data=train_data,
        nfolds=nfolds,
        seed=args.seed,
        transform=pre_transforms_train,
        cache_dir=args.cache_dir,
    )

    train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
    val_dss = [cvdataset.get_dataset(folds=i, transform=pre_transforms_val) for i in range(nfolds)]

    train_loaders = [
        ThreadDataLoader(
            train_dss[i],
            shuffle=True,
            num_workers=args.num_workers,
            batch_size=1,
        )
        for i in folds
    ]

    val_loaders = [
        ThreadDataLoader(
            val_dss[i],
            num_workers=args.num_workers,
            batch_size=1,
        )
        for i in folds
    ]

    return train_loaders, val_loaders  # , test_loader


def get_metrics_loader(args, file_glob="*.nii.gz"):
    labels_dir = args.labels_dir
    predictions_dir = args.predictions_dir
    predictions_glob = os.path.join(predictions_dir, file_glob)
    test_predictions = sorted(glob.glob(predictions_glob))
    test_datalist = []

    for pred_file_name in test_predictions:
        logger.info(f"{pred_file_name=}")
        assert os.path.exists(pred_file_name)
        file_name = get_filename_without_extensions(pred_file_name)
        label_file_name = os.path.join(labels_dir, f"{file_name}{file_glob[1:]}")
        assert os.path.exists(label_file_name)
        logger.info(f"{label_file_name=}")
        test_datalist.append({CommonKeys.LABEL: label_file_name, CommonKeys.PRED: pred_file_name})

    test_datalist = test_datalist[0 : args.limit] if args.limit else test_datalist
    total_l = len(test_datalist)
    assert total_l > 0

    logger.info("{} :: Total Records used for Dataloader is: {}".format(args.gpu, total_l))

    return test_datalist


def get_metrics_transforms(device, labels, args):
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    t = [
        InitLoggerd(loglevel=loglevel, no_log=args.no_log, log_dir=args.output_dir),
        LoadImaged(
            keys=["pred", "label"],
            reader="ITKReader",
            image_only=False,
        ),
        ToDeviced(keys=["pred", "label"], device=device),
        EnsureChannelFirstd(keys=["pred", "label"]),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(False, False),
            to_onehot=(len(labels), len(labels)),
        ),
    ]

    return Compose(t)
