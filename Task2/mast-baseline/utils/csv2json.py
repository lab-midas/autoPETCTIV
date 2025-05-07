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





if __name__ == "__main__":
    # Path to your directory with the data curation
    dst_basedir = "" 
    parse_csv()


