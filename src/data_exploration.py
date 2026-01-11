# src/data_exploration.py
import matplotlib.pyplot as plt
import numpy as np
from data_preprocess import load_main_data, load_recovery_data


def steering_distribution_plot(df, title="Steering Angle Distribution", bins=25):
    plt.figure(figsize=(8, 4))
    plt.hist(df["steering"].astype(float), bins=bins)
    plt.title(title)
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def compare_main_vs_recovery(main_df, recovery_df, bins=25):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(main_df["steering"].astype(float), bins=bins)
    plt.title("Track 1 Main Driving")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(recovery_df["steering"].astype(float), bins=bins)
    plt.title("Recovery Driving")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def dataset_size_summary(df, name="Dataset"):
    print(f"\n--- {name} ---")
    print("Total samples:", len(df))
    print("Unique center images:", df["center"].nunique())
    print("Unique left images:", df["left"].nunique())
    print("Unique right images:", df["right"].nunique())


def near_zero_percentage(df, zero_thresh=0.03):
    angles = df["steering"].astype(float).values
    return (np.mean(np.abs(angles) < zero_thresh) * 100.0)


def downsample_near_zero(df, zero_thresh=0.03, keep_prob=0.10, seed=42):
    """
    Keep only a fraction of near-zero steering angles to reduce the big spike at 0.
    """
    rng = np.random.default_rng(seed)
    angles = df["steering"].astype(float).values

    mask_zero = np.abs(angles) < zero_thresh
    keep_mask = np.ones(len(df), dtype=bool)

    # randomly keep only some of the near-zero rows
    keep_mask[np.where(mask_zero)[0]] = rng.random(np.sum(mask_zero)) < keep_prob
    return df.loc[keep_mask].reset_index(drop=True)


def balance_by_bins(df, n_bins=25, max_per_bin=350, seed=42):
    """
    Cap number of samples per steering bin to flatten distribution.
    """
    rng = np.random.default_rng(seed)
    angles = df["steering"].astype(float).values

    hist, bin_edges = np.histogram(angles, bins=n_bins)
    keep_idx = []

    for b in range(n_bins):
        left, right = bin_edges[b], bin_edges[b + 1]
        idx = np.where((angles >= left) & (angles < right))[0]

        if len(idx) > max_per_bin:
            idx = rng.choice(idx, size=max_per_bin, replace=False)

        keep_idx.extend(idx.tolist())

    balanced = df.iloc[keep_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


if __name__ == "__main__":
    main_df = load_main_data()
    recovery_df = load_recovery_data()

    # Combine
    combined_df = main_df.copy()._append(recovery_df, ignore_index=True)

    # ---- BEFORE ----
    dataset_size_summary(main_df, "Track 1 Main")
    dataset_size_summary(recovery_df, "Track 1 Recovery")
    dataset_size_summary(combined_df, "Combined (Main + Recovery)")

    print(f"Near-zero % (|angle|<0.03) BEFORE: {near_zero_percentage(combined_df, 0.03):.2f}%")

    steering_distribution_plot(combined_df, "Combined Steering Distribution (BEFORE)", bins=50)
    compare_main_vs_recovery(main_df, recovery_df, bins=50)

    # ---- STEP 1: Near-zero downsample ----
    # Tuning knobs:
    ZERO_THRESH = 0.03
    KEEP_PROB = 0.10  # try 0.08–0.15 if needed

    combined_nz = downsample_near_zero(combined_df, zero_thresh=ZERO_THRESH, keep_prob=KEEP_PROB)

    dataset_size_summary(combined_nz, "Combined after Near-Zero Downsample")
    print(f"Near-zero % AFTER near-zero downsample: {near_zero_percentage(combined_nz, ZERO_THRESH):.2f}%")
    steering_distribution_plot(combined_nz, "After Near-Zero Downsample", bins=50)

    # ---- STEP 2: Bin balancing ----
    N_BINS = 25
    MAX_PER_BIN = 350  # adjust depending on dataset size

    combined_bal = balance_by_bins(combined_nz, n_bins=N_BINS, max_per_bin=MAX_PER_BIN)

    dataset_size_summary(combined_bal, "Combined after Bin Balancing")
    print(f"Near-zero % FINAL: {near_zero_percentage(combined_bal, ZERO_THRESH):.2f}%")
    steering_distribution_plot(combined_bal, "Final Steering Distribution (After Balancing)", bins=50)

    print("\nNext step: train using `combined_bal` (NOT the raw combined_df).")
