# exploration: visuals and summaries
import matplotlib.pyplot as plt
from data_preprocess import load_data

def steering_distribution_plot(df):
    plt.figure(figsize=(8, 4))
    plt.hist(df['steering'], bins=25)
    plt.title("Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()

    print("Total samples:", len(df))
    print(df.head())

    steering_distribution_plot(df)

