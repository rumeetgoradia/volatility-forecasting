# Progress tracking for resumable model training

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class ProgressTracker:
    def __init__(self, progress_file: str = "outputs/progress/training_progress.json"):
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress = self._load()

    def _load(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def is_completed(self, model_type: str, instrument: str) -> bool:
        if model_type not in self.progress:
            return False
        if instrument not in self.progress[model_type]:
            return False
        return self.progress[model_type][instrument].get("status") == "completed"

    def mark_completed(self, model_type: str, instrument: str, metrics: dict):
        if model_type not in self.progress:
            self.progress[model_type] = {}

        # ensure all metrics are JSON-serializable
        metrics = {k: float(v) for k, v in metrics.items()}

        self.progress[model_type][instrument] = {
            "status": "completed",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def mark_in_progress(self, model_type: str, instrument: str):
        if model_type not in self.progress:
            self.progress[model_type] = {}

        self.progress[model_type][instrument] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def mark_failed(self, model_type: str, instrument: str, error: str):
        if model_type not in self.progress:
            self.progress[model_type] = {}

        self.progress[model_type][instrument] = {
            "status": "failed",
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def get_pending(self, model_type: str, all_instruments: List[str]) -> List[str]:
        if model_type not in self.progress:
            return all_instruments

        pending = []
        for instrument in all_instruments:
            if not self.is_completed(model_type, instrument):
                pending.append(instrument)

        return pending

    def get_failed(self, model_type: str) -> List[str]:
        if model_type not in self.progress:
            return []

        failed = []
        for instrument, info in self.progress[model_type].items():
            if info.get("status") == "failed":
                failed.append(instrument)

        return failed

    def clear(self, model_type: Optional[str] = None):
        if model_type:
            if model_type in self.progress:
                del self.progress[model_type]
        else:
            self.progress = {}
        self._save()

    def summary(self) -> str:
        if not self.progress:
            return "No training progress found"

        lines = ["Training Progress:"]

        for model_type, instruments in self.progress.items():
            lines.append(f"\n{model_type.upper()}:")

            completed = [
                i
                for i, info in instruments.items()
                if info.get("status") == "completed"
            ]
            in_progress = [
                i
                for i, info in instruments.items()
                if info.get("status") == "in_progress"
            ]
            failed = [
                i for i, info in instruments.items() if info.get("status") == "failed"
            ]

            if completed:
                lines.append(f"  Completed: {len(completed)}")
                for inst in completed:
                    metrics = instruments[inst].get("metrics", {})
                    rmse = metrics.get("rmse", "N/A")
                    if isinstance(rmse, float):
                        lines.append(f"    {inst}: RMSE={rmse:.6f}")
                    else:
                        lines.append(f"    {inst}: RMSE={rmse}")

            if in_progress:
                lines.append(f"  In Progress: {', '.join(in_progress)}")

            if failed:
                lines.append(f"  Failed: {len(failed)}")
                for inst in failed:
                    error = instruments[inst].get("error", "Unknown error")
                    lines.append(f"    {inst}: {error}")

        return "\n".join(lines)

    def get_results_dataframe(self):
        import pandas as pd

        results = []
        for model_type, instruments in self.progress.items():
            for instrument, info in instruments.items():
                if info.get("status") == "completed":
                    metrics = info.get("metrics", {})
                    result = {
                        "model": model_type.upper(),
                        "instrument": instrument,
                        "val_rmse": metrics.get("rmse"),
                        "val_mae": metrics.get("mae"),
                        "val_qlike": metrics.get("qlike"),
                        "val_r2": metrics.get("r2"),
                        "n_samples": metrics.get("n_samples"),
                    }
                    results.append(result)

        return pd.DataFrame(results) if results else pd.DataFrame()
