import pandas as pd
from pathlib import Path


def compare_all_models():
    results_path = Path("outputs/results")

    print("MODEL COMPARISON")

    baseline_file = results_path / "baseline_results_scaled.csv"
    neural_file = results_path / "neural_results.csv"
    moe_file = results_path / "moe_results.csv"

    results = {}

    if baseline_file.exists():
        df = pd.read_csv(baseline_file)
        har_rmse = df["val_rmse"].mean()
        har_r2 = df["val_r2"].mean()
        results["HAR-RV"] = {"RMSE": har_rmse, "R2": har_r2}
        print(f"HAR-RV    RMSE: {har_rmse:.6f}, R2: {har_r2:.4f}")

    if neural_file.exists():
        df = pd.read_csv(neural_file)

        lstm_df = df[df["model"] == "LSTM"]
        if len(lstm_df) > 0:
            lstm_rmse = lstm_df["val_rmse"].mean()
            lstm_r2 = lstm_df["val_r2"].mean()
            results["LSTM"] = {"RMSE": lstm_rmse, "R2": lstm_r2}
            print(f"LSTM      RMSE: {lstm_rmse:.6f}, R2: {lstm_r2:.4f}")

        tcn_df = df[df["model"] == "TCN"]
        if len(tcn_df) > 0:
            tcn_rmse = tcn_df["val_rmse"].mean()
            tcn_r2 = tcn_df["val_r2"].mean()
            results["TCN"] = {"RMSE": tcn_rmse, "R2": tcn_r2}
            print(f"TCN       RMSE: {tcn_rmse:.6f}, R2: {tcn_r2:.4f}")

    if moe_file.exists():
        df = pd.read_csv(moe_file)
        moe_rmse = df["val_rmse"].mean()
        moe_r2 = df["val_r2"].mean()
        results["MoE"] = {"RMSE": moe_rmse, "R2": moe_r2}
        print(f"MoE       RMSE: {moe_rmse:.6f}, R2: {moe_r2:.4f}")

    if len(results) > 1:
        print("\nRanking by RMSE:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]["RMSE"])
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"  {i}. {name}: {metrics['RMSE']:.6f}")

        if "MoE" in results:
            best_individual = sorted_models[0][0]
            best_rmse = sorted_models[0][1]["RMSE"]
            moe_rmse = results["MoE"]["RMSE"]

            improvement = (best_rmse - moe_rmse) / best_rmse * 100

            print(f"\nMoE vs Best Individual ({best_individual}):")
            if improvement > 0:
                print(f"  MoE is {improvement:.2f}% better")
            elif improvement < 0:
                print(f"  MoE is {abs(improvement):.2f}% worse")
            else:
                print(f"  MoE is equivalent")

            if improvement < -5:
                print(f"\n  ISSUE: MoE significantly worse than best individual")
                print(f"  This means the gating network is hurting performance")
            elif improvement < 1:
                print(f"\n  ISSUE: MoE provides minimal benefit")
                print(f"  Gating network just learned to pick LSTM (best individual)")
            else:
                print(f"\n  SUCCESS: MoE improves over individual experts")


if __name__ == "__main__":
    compare_all_models()
