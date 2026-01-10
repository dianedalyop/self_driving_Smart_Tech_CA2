# src/data_exploration.py
import matplotlib.pyplot as plt
from data_preprocess import load_main_data, load_recovery_data


def steering_distribution_plot(df, title="Steering Angle Distribution"):
    plt.figure(figsize=(8, 4))
    plt.hist(df["steering"], bins=25)
    plt.title(title)
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def compare_main_vs_recovery(main_df, recovery_df):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(main_df["steering"], bins=25)
    plt.title("Track 1 Main Driving")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(recovery_df["steering"], bins=25)
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


if __name__ == "__main__":
    main_df = load_main_data()
    recovery_df = load_recovery_data()

    # 1) Count how many images
    dataset_size_summary(main_df, "Track 1 Main")
    dataset_size_summary(recovery_df, "Track 1 Recovery")

    # Optional: combined totals
    combined_df = main_df.copy()
    combined_df = combined_df._append(recovery_df, ignore_index=True)
    dataset_size_summary(combined_df, "Combined (Main + Recovery)")

    # 2) Plots
    steering_distribution_plot(combined_df, "Combined Steering Distribution")
    compare_main_vs_recovery(main_df, recovery_df)
