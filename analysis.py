import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_agent_performance(file_path="logs/SAC/monitor.csv"):
    # Load the data, skipping the metadata header line
    try:
        df = pd.read_csv(file_path, skiprows=1)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. Basic Calculations
    df["reward_density"] = df["r"] / df["l"]  # Reward per step (r/l)
    df["rolling_r"] = df["r"].rolling(window=50).mean()
    df["rolling_l"] = df["l"].rolling(window=50).mean()

    print("--- AGENT STATISTICS ---")
    print(f"Total Episodes: {len(df)}")
    print(f"Mean Reward: {df['r'].mean():.2f}")
    print(f"Mean Length: {df['l'].mean():.2f}")
    print(f"Correlation (r vs l): {df['r'].corr(df['l']):.4f}")

    # 2. Plotting: Learning Progress
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df["r"], alpha=0.3, color="blue", label="Raw Reward")
    plt.plot(
        df.index, df["rolling_r"], color="red", linewidth=2, label="Rolling Avg (50)"
    )
    plt.title("Episode Reward ($r$) over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # 3. Plotting: Reward vs Length (Farming Check)
    plt.subplot(1, 2, 2)
    plt.scatter(df["l"], df["r"], alpha=0.4, c=df.index, cmap="viridis")
    plt.title("Reward ($r$) vs Length ($l$)")
    plt.xlabel("Episode Length (Steps)")
    plt.ylabel("Total Reward")
    plt.colorbar(label="Episode #")

    plt.tight_layout()
    plt.savefig("learning_curve.png")
    print("Saved learning_curve.png")

    # 4. Reward Density Analysis
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["reward_density"].rolling(window=100).mean(), color="green")
    plt.title("Reward Density ($r/l$) - Efficiency Metric")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward per Step")
    plt.savefig("reward_density.png")
    print("Saved reward_density.png")


if __name__ == "__main__":
    analyze_agent_performance()
