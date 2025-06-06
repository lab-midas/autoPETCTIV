import json
import os
from glob import glob
import SimpleITK as sitk
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)




def check_topology_class(path, id):
    df = pd.read_csv(os.path.join(path, "inputsTr", f"{id}.csv"))
    num_lesions = len(df)
    img_bl_ids = df["img_id_bl"].unique()
    img_fu_ids = df["img_id_fu"].unique()

    # for each bl_id load the mask
    values_bl = []
    for bl_id in img_bl_ids:
        mask_file = os.path.join(
            path, "inputsTr", f"{id}_BL_mask_BL_img_{bl_id:02}.nii.gz"
        )
        mask_input = sitk.ReadImage(mask_file)
        mask_array = sitk.GetArrayFromImage(mask_input)

        # get unique values in the mask
        unique_values = np.unique(mask_array)
        # remove 0 from the list
        unique_values = unique_values[unique_values != 0]
        values_bl = values_bl + unique_values.tolist()

    values_fu = []
    for fu_id in img_fu_ids:
        mask_file = os.path.join(
            path, "targetsTr", f"{id}_FU_mask_FU_img_{fu_id:02}.nii.gz"
        )
        mask_input = sitk.ReadImage(mask_file)
        mask_array = sitk.GetArrayFromImage(mask_input)

        # get unique values in the mask
        unique_values = np.unique(mask_array)
        # remove 0 from the list
        unique_values = unique_values[unique_values != 0]
        values_fu = values_fu + unique_values.tolist()

    print(
        f"Length for {id}: Number of lesions in the csv file ({len(df)}), Number of lesions BL: ({len(values_bl )}) , Number of lesions FU: ({len(values_fu )})"
    )
    # check if len dataframe is equal to the number of unique values in the masks
    if len(df) != len(set(values_bl + values_fu)):
        print(
            f"Error for {id}: Number of lesions in the csv file ({len(df)}) is not equal to the number of unique values in the masks ({len(set(values_bl + values_fu))})"
        )
        return id
    already_assigned = []
    # iterate over df['lesions_id'] and set topology_class
    for lesion_id in df["lesion_id"].values:
        if lesion_id in already_assigned:
            continue
        print(lesion_id)
        if not lesion_id in values_fu:
            # check how often the cog_fu is in df["cog_fu"]
            cog_fu = df[df["lesion_id"] == lesion_id]["cog_fu"].values[0]
            df_cog_fu = df[df["cog_fu"] == cog_fu]
            if len(df_cog_fu) > 1:
                # check if of the lesion_ids is in values_fu
                ids_with_cog_fu = df_cog_fu["lesion_id"].values
                id_with_cog_in_fu = [id_ for id_ in ids_with_cog_fu if id_ in values_fu]
                if len(id_with_cog_in_fu) > 1:
                    print(
                        "Error: More than one lesion_id with the same cog_fu in fu_image"
                    )
                else:
                    merged_into = id_with_cog_in_fu[0]

                    # set topology class in df for lesion_id== ids_with_cog_fu
                    df.loc[df["lesion_id"].isin(ids_with_cog_fu), "topology_class"] = (
                        "MERGING"
                    )

                    # set merged_into in df for lesion_id== ids_with_cog_fu
                    df.loc[df["lesion_id"].isin(ids_with_cog_fu), "merged_into"] = (
                        merged_into
                    )
                    already_assigned = already_assigned + ids_with_cog_fu.tolist()

            elif len(df_cog_fu) == 1 or np.isnan(cog_fu):
                df.loc[df["lesion_id"] == lesion_id, "topology_class"] = (
                    'DISAPPEARING'
                )
                already_assigned.append(lesion_id)

        elif not lesion_id in values_bl:
            # check how often the cog_bl is in df["cog_bl"]
            cog_bl = df[df["lesion_id"] == lesion_id]["cog_bl"].values[0]
            df_cog_bl = df[df["cog_bl"] == cog_bl]
            if len(df_cog_bl) > 1:
                # check if of the lesion_ids is in values_bl
                ids_with_cog_bl = df_cog_bl["lesion_id"].values
                id_with_cog_in_bl = [id_ for id_ in ids_with_cog_bl if id_ in values_bl]
                if len(id_with_cog_in_bl) > 1:
                    print(
                        "Error: More than one lesion_id with the same cog_bl in bl_image"
                    )
                else:
                    split_from = id_with_cog_in_bl[0]

                    # set topology class in df for lesion_id== ids_with_cog_fu
                    df.loc[df["lesion_id"].isin(ids_with_cog_bl), "topology_class"] = (
                        "SPLITTING"
                    )

                    # set merged_into in df for lesion_id== ids_with_cog_fu
                    df.loc[df["lesion_id"].isin(ids_with_cog_bl), "split_from"] = split_from

                    already_assigned = already_assigned + ids_with_cog_bl.tolist()

            elif len(df_cog_bl) == 1:
                df.loc[df["lesion_id"] == lesion_id, "topology_class"] = (
                    "NEWLYAPPEARING"
                )
                already_assigned.append(lesion_id)

        elif lesion_id in values_bl and lesion_id in values_fu:
            df.loc[df["lesion_id"] == lesion_id, "topology_class"] = (
                "UNCHANGED"
            )
            already_assigned.append(lesion_id)

    if not "merged_into" in df.columns:
        df["merged_into"] = None

    return df


path = '' ### add your path here ###

pids_csv = glob(os.path.join(path, "inputsTr", "*.csv"))
pids = [os.path.basename(pid)[: -len(".csv")] for pid in pids_csv]
pids.sort()

for i in range(len(pids)):
    id = pids[i]
    print(id)
    df = pd.read_csv(os.path.join(path, "inputsTr", f"{id}.csv"))
    topology = check_topology_class(path, id)
    if type(topology) == str:
        print(f"Error for {id}: {topology}")
    else:
        # save topology to csv
        topology.to_csv(os.path.join(path, f"{id}.csv"), index=False)
