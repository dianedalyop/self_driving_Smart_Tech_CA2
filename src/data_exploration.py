# exploration: visuals and summaries
import matplotlib.pyplot as plt
from data_preprocess import load_data
import matplotlib.pyplot as plt
from data_preprocess import load_data, load_main_data, load_recovery_data



def steering_distribution_plot(df):
    plt.figure(figsize=(8, 4))
    plt.hist(df['steering'], bins=25)
    plt.title("Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def compare_main_vs_recovery(main_df, recovery_df):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(main_df['steering'], bins=25)
    plt.title("Track 1 Main Driving")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(recovery_df['steering'], bins=25)
    plt.title("Recovery Driving")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    df = load_data()
    main_df = load_main_data()
    recovery_df = load_recovery_data()

    print("Total samples:", len(df))
    print("Main driving samples:", len(main_df))
    print("Recovery samples:", len(recovery_df))

    steering_distribution_plot(df)
    compare_main_vs_recovery(main_df, recovery_df)

