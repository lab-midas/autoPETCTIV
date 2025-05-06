import json
import os
import math
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


def print_number_of_cog_nans():
    cog_b = []
    cog_f = []
    cog_p = []

    csv_files = glob(os.path.join(dst_basedir, "inputsTr", "*.csv"))
    sorted(csv_files)

    for csv_file in csv_files:
        # Load the CSV file
        try:
            df = pd.read_csv(csv_file)

            # Access a specific column
            cog_b += list(df["cog_bl"])
            cog_p += list(df["cog_propagated"])
            cog_f += list(df["cog_fu"])
        except:
            print("Failed for {}".format(csv_file))

    print(
        f"None count cog_bl = {sum(1 for x in cog_b if isinstance(x, float) and math.isnan(x))}/{len(cog_b)}"
    )
    print(
        f"None count cog_p = {sum(1 for x in cog_p if isinstance(x, float) and math.isnan(x))}/{len(cog_p)}"
    )
    print(
        f"None count cog_fu = {sum(1 for x in cog_f if isinstance(x, float) and math.isnan(x))}/{len(cog_f)}"
    )
    print(
        f"None count cog_fu and cog_p = {sum(1 for i, x in enumerate(cog_f) if isinstance(x, float) and isinstance(cog_p[i], float) and math.isnan(cog_p[i]) and math.isnan(x))}/{len(cog_f)}"
    )


def print_multiple_time_points():
    csv_files = glob(os.path.join(dst_basedir, "inputsTr", "*.csv"))
    sorted(csv_files)

    id_tuples = []

    for csv_file in csv_files:
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Access a specific column
        id_tuples += [
            (id_bl, list(df["img_id_fu"])[ind])
            for ind, id_bl in enumerate(list(df["img_id_bl"]))
        ]


    from collections import Counter

    tuple_counts = Counter(id_tuples)

    # Convert the counts to a list of sets and associated counts
    result = list(tuple_counts.items())

    print(result)


def get_points_dict(points):
    return {
        "name": "Points of interest",
        "type": "Multiple points",
        "points": points,
        "version": {"major": 1, "minor": 0},
    }


def parse_csv():
    csv_files = glob(os.path.join(dst_basedir, "inputsTr", "*.csv"))
    sorted(csv_files)


    for csv_file in csv_files:
        print(csv_file)
        # Load the CSV file
        df = pd.read_csv(csv_file)

        num_bl = max(list(df["img_id_bl"])) + 1
        num_fu = max(list(df["img_id_fu"])) + 1

        bl_points = [[] for _ in range(num_bl)]
        fu_points = [[] for _ in range(num_fu)]

        for index, row in df.iterrows():
            # print(f"Index: {index}, Data: {row['lesion_id']}")  # Replace
            bl_points[row["img_id_bl"]].append(
                {
                    "name": str(row["lesion_id"]),
                    "point": [float(pax) for pax in row["cog_bl"].split(" ")],
                }
            )

            point_fu = (
                row["cog_fu"]
                if isinstance(row["cog_propagated"], float)
                and math.isnan(row["cog_propagated"])
                else row["cog_propagated"]
            )

            fu_points[row["img_id_fu"]].append(
                {
                    "name": str(row["lesion_id"]),
                    "point": [float(pax) for pax in point_fu.split(" ")],
                }
            )

        for bl_id, bl_point in enumerate(bl_points):
            json.dump(
                get_points_dict(bl_point),
                open(csv_file[: -len(".csv")] + f'_BL_{"%02d" % (bl_id)}.json', "w"),
                sort_keys=True,
                indent=4,
                cls=NumpyEncoder,
            )

        for fu_id, fu_point in enumerate(fu_points):
            json.dump(
                get_points_dict(fu_point),
                open(csv_file[: -len(".csv")] + f'_FU_{"%02d" % (fu_id)}.json', "w"),
                sort_keys=True,
                indent=4,
                cls=NumpyEncoder,
            )





dst_basedir = r"C:\Users\tkohlbrandt\Documents\MAST\harmonisied_test" #path to your directory with the data curation
parse_csv()


