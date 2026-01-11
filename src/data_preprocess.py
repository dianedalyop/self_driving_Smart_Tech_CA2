
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths to CSV logs
TRACK1_MAIN = "data/track1_main/Udacity_sim_Track1_rec/driving_log.csv"
TRACK1_RECOVERY = "data/track1_recovery/Udacity_sim_track1_recovery/driving_log.csv"

columns_name = [
    "center",
    "left",
    "right",
    "steering",
    "throttle",
    "brake",
    "speed",
]


# ------------------------------------------------------------
# Image preprocessing (NVIDIA format)
# ------------------------------------------------------------
def preprocess_image(img):
    if img is None:
        raise ValueError("None image received in preprocess_image")

    # Cropping (top 50px, bottom 20px off)
    img = img[50:-20, :, :]

    # NVIDIA expected input size (width=200, height=66)
    img = cv2.resize(img, (200, 66))

    # Convert BGR OpenCV -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalizing to [-0.5, 0.5]
    img = img / 255.0 - 0.5
    return img



def load_data():
    df1 = pd.read_csv(TRACK1_MAIN, header=None, names=columns_name)
    df2 = pd.read_csv(TRACK1_RECOVERY, header=None, names=columns_name)

    df = pd.concat([df1, df2], ignore_index=True)

    print("Total number of samples:", len(df))
    print("Columns:", df.columns)

    print("Table rows:")
    print(df.head())

    return df


def load_main_data():
    return pd.read_csv(TRACK1_MAIN, header=None, names=columns_name)


def load_recovery_data():
    return pd.read_csv(TRACK1_RECOVERY, header=None, names=columns_name)



def _fix_img_path(csv_path: str, img_path: str) -> str:
   
    if not isinstance(img_path, str):
        return img_path

    img_path = img_path.strip()
    base_dir = os.path.dirname(csv_path)

   
    if os.path.isabs(img_path):
        return os.path.normpath(img_path)

    
    return os.path.normpath(os.path.join(base_dir, img_path))


def _load_udacity_csv(csv_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path, header=None, names=columns_name)

    df["center"] = df["center"].apply(lambda p: _fix_img_path(csv_path, p))
    df["left"] = df["left"].apply(lambda p: _fix_img_path(csv_path, p))
    df["right"] = df["right"].apply(lambda p: _fix_img_path(csv_path, p))

    df["steering"] = df["steering"].astype(float)

    return df[["center", "left", "right", "steering"]]


def downsample_near_zero(df: pd.DataFrame, threshold: float = 0.03, keep_prob: float = 0.10, seed: int = 42):
    
    rng = np.random.default_rng(seed)

    mask_zero = df["steering"].abs() < threshold
    zero_idx = df.index[mask_zero].to_numpy()

    keep_mask = np.ones(len(df), dtype=bool)

    
    drop_draw = rng.random(len(zero_idx)) > keep_prob
    drop_idx = zero_idx[drop_draw]

    keep_mask[df.index.get_indexer(drop_idx)] = False
    return df.loc[keep_mask].reset_index(drop=True)


def balance_by_bins(df: pd.DataFrame, bins: int = 25, max_per_bin: int = 300, seed: int = 42):
    
    rng = np.random.default_rng(seed)

    hist, edges = np.histogram(df["steering"], bins=bins)

    kept_parts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = df[(df["steering"] >= lo) & (df["steering"] < hi)]

        if len(in_bin) <= max_per_bin:
            kept_parts.append(in_bin)
        else:
            sampled_idx = rng.choice(in_bin.index.to_numpy(), size=max_per_bin, replace=False)
            kept_parts.append(df.loc[sampled_idx])

    out = pd.concat(kept_parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def build_training_df(
    track1_main_csv: str = TRACK1_MAIN,
    track1_recovery_csv: str = TRACK1_RECOVERY,
    near_zero_threshold: float = 0.03,
    near_zero_keep_prob: float = 0.10,
    bins: int = 25,
    max_per_bin: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    
    main_df = _load_udacity_csv(track1_main_csv)
    rec_df = _load_udacity_csv(track1_recovery_csv)

    df = pd.concat([main_df, rec_df], ignore_index=True)

   
    df = downsample_near_zero(df, threshold=near_zero_threshold, keep_prob=near_zero_keep_prob, seed=seed)

    
    df = balance_by_bins(df, bins=bins, max_per_bin=max_per_bin, seed=seed)

    return df



if __name__ == "__main__":
    df_raw = load_data()
    df_bal = build_training_df()

    print("\n--- Balanced dataset summary ---")
    print("Balanced samples:", len(df_bal))
    print("Near-zero %:", (df_bal["steering"].abs() < 0.03).mean() * 100)
    print(df_bal.head())
