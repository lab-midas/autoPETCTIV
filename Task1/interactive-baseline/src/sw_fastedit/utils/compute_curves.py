import numpy as np
import os
from matplotlib import pyplot as plt

root_dir = 'demo_json/Ts_clicks_center/val_metrics/'
root_dir = 'output_edt/val_metrics/'
npz_files = sorted(os.listdir(root_dir))
npz_files = [os.path.join(root_dir, el) for el in npz_files]

dscs = np.zeros((1, 10))
fpvs = np.zeros((1, 10))
fnvs = np.zeros((1, 10))
def plot_curves(dscs_means, fpvs_means, fnvs_means, fn):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each array in a subplot
    axs[0].plot(dscs_means, label="DSCs", color="blue", marker="o")
    #axs[0].fill_between(np.arange(10), np.subtract(dscs_means, dscs_std), np.add(dscs_means, dscs_std), alpha=0.2)

    axs[0].set_title("DSC Values")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(fpvs_means, label="FPV", color="green", marker="x")
    #axs[1].fill_between(np.arange(10), np.subtract(fpvs_means, fpvs_std), np.add(fpvs_means, fpvs_std), alpha=0.2)

    axs[1].set_title("FPV Values")
    axs[1].set_xlabel("Index")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(fnvs_means, label="FNV", color="red", marker="s")
    #axs[2].fill_between(np.arange(10), np.subtract(fnvs_means, fnvs_std), np.add(fnvs_means, fnvs_std), alpha=0.2)

    axs[2].set_title("FNV Values")
    axs[2].set_xlabel("Index")
    axs[2].grid(True)
    axs[2].legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.savefig(fn)

for i, npz_file in enumerate(npz_files):
    x = np.load(npz_file)
    dsc = np.array(x['dsc'])
    fpv = np.array(x['fpv'])
    fnv = np.array(x['fnv'])
    dscs = np.concatenate([dscs, np.expand_dims(dsc, axis=0)])
    fpvs = np.concatenate([fpvs, np.expand_dims(fpv, axis=0)])
    fnvs = np.concatenate([fnvs, np.expand_dims(fnv, axis=0)])

    #plot_curves(dsc, fpv, fnv, f'{i}.png')
final_dice = np.array([el[-1] for el in dscs])
final_fpv = np.array([el[-1] for el in fpvs])
final_fpn = np.array([el[-1] for el in fnvs])
data = [final_dice, final_fpv, final_fpn]
labels = ['Final Dice', 'Final FPV', 'Final FPN']

plt.figure(figsize=(6, 4))
plt.boxplot(final_dice, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Box Plot of Final Dice")
plt.ylabel("Values")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("final_dice_boxplot.png")
plt.close()

# Plot and save for final_fpv
plt.figure(figsize=(6, 4))
plt.boxplot(final_fpv, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
plt.title("Box Plot of Final FPV")
plt.ylabel("Values")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("final_fpv_boxplot.png")
plt.close()

# Plot and save for final_fpn
plt.figure(figsize=(6, 4))
plt.boxplot(final_fpn, patch_artist=True, boxprops=dict(facecolor="lightcoral"))
plt.title("Box Plot of Final FPN")
plt.ylabel("Values")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("final_fpn_boxplot.png")
plt.close()
exit()


#dscs_means = np.median(dscs, axis=0)
#fpvs_means = np.median(fpvs, axis=0)
#fnvs_means = np.median(fnvs, axis=0)

dscs_std = np.std(dscs, axis=0)
fpvs_std = np.std(fpvs, axis=0)
fnvs_std = np.std(fnvs, axis=0)



