from monai.networks.nets.dynunet import DynUNet
from monai.inferers import SlidingWindowInferer
import torch
from sw_fastedit.data import get_test_loader, get_pre_transforms_val_as_list, get_click_transforms_json, get_post_transforms
from monai.transforms import Compose
from sw_fastedit.api import init
from sw_fastedit.utils.argparser import parse_args, setup_environment_and_adapt_args
from monai.metrics import DiceMetric
import numpy as np
import os
from scipy import integrate
from collections import OrderedDict
import pandas as pd

# Example:
# python src/simplified_inference.py -a -i PET-volumes/ -o ./output/  -c ./cache -ta -e 800 --dont_check_output_dir --resume_from model/151_best_0.8534.pt --eval_only --json_dir demo_json/ --no_log --no_data --loop
# Directory Structure:
# PET-volumes/
# -----------volume_1.nii.gz
# -----------volume_2.nii.gz 
# ...
#
# demo_json/
# -----------volume_1_clicks.json
# -----------volume_2_clicks.json
# ...
#

def simplified_predict(input_folder, output_folder, json_folder, docker):
    args = parse_args()
    if docker:
        args.input_dir = input_folder
        args.json_dir = json_folder
        args.output_dir = output_folder
    setup_environment_and_adapt_args(args, docker)

    init(args)
    if not docker:
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        metric = OrderedDict()
        metric['CaseName'] = []
        # 6 Metrics
        metric['DSC_AUC'] = []
        metric['FPV_AUC'] = []
        metric['FNV_AUC'] = []
        metric['DSC_Final'] = []
        metric['FPV_Final'] = []
        metric['FNV_Final'] = []

    network = DynUNet(
        spatial_dims=3,
        in_channels=3, # image + fg + bg = 3
        out_channels=2, # len(labels) = fg + bg = 2
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )

    network.load_state_dict(torch.load(args.resume_from)["net"])

    sw_params = {
        "roi_size": (128, 128, 128),
        "mode": "gaussian",
        "cache_roi_weight_map": True,
    }
    eval_inferer = SlidingWindowInferer(sw_batch_size=1, overlap=0.25, **sw_params)

    loop = args.loop
    n_clicks = 10
    device = torch.device('cuda:0') 
    pre_transforms_val = Compose(get_pre_transforms_val_as_list(args.labels, device, args))
    if loop:
        args.save_pred = False
    if docker:
        args.save_pred = True
    post_transform = get_post_transforms(args.labels, save_pred=args.save_pred, output_dir=args.output_dir, pretransform=pre_transforms_val, docker=docker)

    val_loader = get_test_loader(args, pre_transforms_val)
    network.eval()
    network.to(device)



    for data in val_loader:
        data['image'] = data['image'].to(device)[0]
        if not docker:
            data['label'] = data['label'].to(device)[0]
        if loop: # For offline evaluation
            if not docker:
                val_metrics = {'dsc': [], 'fpv': [], 'fnv': []}
            for i in range(n_clicks+1):
                click_transforms = get_click_transforms_json(device, args, n_clicks=i)
                img = data['image']
                img_clicks = click_transforms(data)['image'].unsqueeze(0) # img + guidance signals (fg + bg)
                data['image'] = img
                with torch.no_grad():
                    pred = eval_inferer(inputs=img_clicks, network=network)
                    data['pred'] = pred[0]
                    pred = post_transform(data)['pred'].unsqueeze(0)

                    if not docker:
                        dsc = dice_metric(pred.to(device), data['label'].to(device).unsqueeze(0))
                        
                        fpv = torch.sum((data['label'][0].to(device)==0) * (pred[0][1].to(device) == 1))
                        fnv = torch.sum((data['label'][0].to(device)==1) * (pred[0][1].to(device) == 0))

                        dsc = dsc[0][0].cpu().detach().numpy()
                        fpv = fpv.cpu().detach().numpy()
                        fnv = fnv.cpu().detach().numpy()

                        print('Click:', i, 'DSC:', dsc, 'FPV', fpv, 'FNV', fnv)
                        val_metrics['dsc'].append(dsc)
                        val_metrics['fpv'].append(fpv)
                        val_metrics['fnv'].append(fnv)
            if not docker:
                img_fn = data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                os.makedirs(os.path.join(args.json_dir, 'val_metrics'), exist_ok=True)
                np.savez_compressed(os.path.join(args.json_dir, 'val_metrics', img_fn.replace(".nii.gz", ".npz")), 
                                    dsc=val_metrics['dsc'],
                                    fpv=val_metrics['fpv'],
                                    fnv=val_metrics['fnv'])
                print('Saved metrics to', os.path.join(args.json_dir, 'val_metrics', img_fn.replace(".nii.gz", ".npz")))
                
                # Compute interactive metrics
                dsc_auc = integrate.cumulative_trapezoid(val_metrics['dsc'], np.arange(n_clicks + 1))[-1]
                fpv_auc = integrate.cumulative_trapezoid(val_metrics['fpv'], np.arange(n_clicks + 1))[-1]
                fnv_auc = integrate.cumulative_trapezoid(val_metrics['fnv'], np.arange(n_clicks + 1))[-1]

                dsc_final = val_metrics['dsc'][-1]
                fpv_final = val_metrics['fpv'][-1]
                fnv_final = val_metrics['fnv'][-1]

                metric['CaseName'].append(img_fn)
                metric['DSC_AUC'].append(dsc_auc)
                metric['FPV_AUC'].append(fpv_auc)
                metric['FNV_AUC'].append(fnv_auc)

                metric['DSC_Final'].append(dsc_final)
                metric['FPV_Final'].append(fpv_final)
                metric['FNV_Final'].append(fnv_final)

        else:
            click_transforms = get_click_transforms_json(device, args, n_clicks=10)
            data['image'] = click_transforms(data)['image'].unsqueeze(0) # img + guidance signals (fg + bg)
            with torch.no_grad():
                pred = eval_inferer(inputs=data['image'], network=network)
                data['pred'] = pred[0]
                data['pred'] = post_transform(data)['pred']
                print('Prediction Done!')
        
    if loop and not docker:
        metric_df = pd.DataFrame(metric)
        metric_df.to_csv(os.path.join(args.json_dir, 'val_metrics', 'interactive_metrics.csv'), index=False)
        print('Saved interactive metrics to', os.path.join(args.json_dir, 'val_metrics', 'interactive_metrics.csv'))

if __name__ == "__main__":
    simplified_predict(None, None, None, False)


        

