"""Pure NumPy Student Performance Analytics Engine."""

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np


# VTU-style evaluation thresholds
MIN_TOTAL_IA = 20.0   # 40% of 50
MIN_SEE = 35.0        # 35% of 100
MIN_FINAL = 40.0      # 40% of 100


@dataclass
class StudentDataset:
    """Container holding student records as aligned NumPy arrays."""

    sl_no: np.ndarray
    names: np.ndarray
    usns: np.ndarray
    streams: np.ndarray
    ia1: np.ndarray       # out of 50
    ia2: np.ndarray       # out of 50
    total_ia: np.ndarray  # (ia1 + ia2) / 2 -> out of 50
    see: np.ndarray       # out of 100

    # Derived NumPy arrays computed on initialization
    see_reduced: np.ndarray = None
    final_score: np.ndarray = None
    ia_diff: np.ndarray = None
    passed_mask: np.ndarray = None
    fail_reason: np.ndarray = None
    grades: np.ndarray = None
    z_scores: np.ndarray = None
    percentiles: np.ndarray = None
    overall_ranks: np.ndarray = None
    stream_ranks: np.ndarray = None

    def __post_init__(self) -> None:
        # Reduce SEE from 100 to 50 so Final Score = Total IA (50) + SEE/2 (50) = 100
        self.see_reduced = np.round(self.see / 2.0, 1)
        self.final_score = np.round(self.total_ia + self.see_reduced, 1)
        self.ia_diff = self.ia2 - self.ia1

        # Pass / Fail Boolean masks
        ia_pass = self.total_ia >= MIN_TOTAL_IA
        see_pass = self.see >= MIN_SEE
        final_pass = self.final_score >= MIN_FINAL
        self.passed_mask = ia_pass & see_pass & final_pass

        self.fail_reason = np.select(
            condlist=[
                self.passed_mask,
                (~ia_pass) & (~see_pass),
                ~ia_pass,
                ~see_pass,
                ~final_pass,
            ],
            choicelist=[
                "PASS",
                "FAIL (IA & SEE)",
                "FAIL (Low IA)",
                "FAIL (Low SEE)",
                "FAIL (Low Total)",
            ],
            default="FAIL",
        )

        # Grade assignment using np.select
        self.grades = np.select(
            condlist=[
                ~self.passed_mask,
                self.final_score >= 90,
                self.final_score >= 80,
                self.final_score >= 70,
                self.final_score >= 60,
                self.final_score >= 55,
                self.final_score >= 50,
                self.final_score >= 40,
            ],
            choicelist=["F", "O", "A+", "A", "B+", "B", "C", "P"],
            default="F",
        )

        # Z-score standardization & percentiles on final_score
        mean_final = np.mean(self.final_score)
        std_final = np.std(self.final_score)
        self.z_scores = (
            (self.final_score - mean_final) / std_final
            if std_final > 0
            else np.zeros_like(self.final_score)
        )

        # Percentile rank for each student (percentage of students scored <= them)
        sorted_scores = np.sort(self.final_score)
        positions = np.searchsorted(sorted_scores, self.final_score, side="right")
        self.percentiles = np.round((positions / len(self.final_score)) * 100.0, 2)

        # Overall rank (1 = highest final_score; ties broken by SEE then Total IA)
        order = np.lexsort((-self.total_ia, -self.see, -self.final_score))
        self.overall_ranks = np.empty(len(self.final_score), dtype=int)
        self.overall_ranks[order] = np.arange(1, len(self.final_score) + 1)

        # Stream-wise rank
        self.stream_ranks = np.empty(len(self.final_score), dtype=int)
        for stream in np.unique(self.streams):
            idx = np.where(self.streams == stream)[0]
            sub_order = np.lexsort(
                (-self.total_ia[idx], -self.see[idx], -self.final_score[idx])
            )
            self.stream_ranks[idx[sub_order]] = np.arange(1, len(idx) + 1)

    def __len__(self) -> int:
        return len(self.usns)

    def filter_stream(self, stream_code: str) -> "StudentDataset":
        """Return a new StudentDataset filtered to a single stream."""
        mask = self.streams == stream_code.upper()
        if not np.any(mask):
            raise ValueError(f"Unknown stream '{stream_code}'.")
        return StudentDataset(
            sl_no=self.sl_no[mask],
            names=self.names[mask],
            usns=self.usns[mask],
            streams=self.streams[mask],
            ia1=self.ia1[mask],
            ia2=self.ia2[mask],
            total_ia=self.total_ia[mask],
            see=self.see[mask],
        )


def load_dataset(csv_path: str | Path) -> StudentDataset:
    """Load student CSV into aligned NumPy arrays."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    sl_no, names, usns, streams = [], [], [], []
    ia1, ia2, total_ia, see = [], [], [], []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sl_no.append(int(row["Sl No"]))
            names.append(row["Name"])
            usns.append(row["USN"])
            streams.append(row["Stream"])
            ia1.append(float(row["IA 1 Marks"]))
            ia2.append(float(row["IA 2 Marks"]))
            total_ia.append(float(row["Total IA"]))
            see.append(float(row["SEE Marks"]))

    return StudentDataset(
        sl_no=np.array(sl_no, dtype=int),
        names=np.array(names, dtype=str),
        usns=np.array(usns, dtype=str),
        streams=np.array(streams, dtype=str),
        ia1=np.array(ia1, dtype=float),
        ia2=np.array(ia2, dtype=float),
        total_ia=np.array(total_ia, dtype=float),
        see=np.array(see, dtype=float),
    )


def exam_summary_stats(ds: StudentDataset) -> dict[str, dict[str, float]]:
    """Compute mean, median, std, min, max, Q1, Q3, P90 for each metric."""
    metrics = {
        "IA 1 (50)": ds.ia1,
        "IA 2 (50)": ds.ia2,
        "Total IA (50)": ds.total_ia,
        "SEE (100)": ds.see,
        "Final Score (100)": ds.final_score,
    }
    stats = {}
    for label, arr in metrics.items():
        q25, q75, p90 = np.percentile(arr, [25, 75, 90])
        stats[label] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(q25),
            "q75": float(q75),
            "p90": float(p90),
        }
    return stats


def stream_comparison(ds: StudentDataset) -> list[dict]:
    """Compare performance metrics across streams using Boolean masking."""
    results = []
    for stream in sorted(np.unique(ds.streams)):
        mask = ds.streams == stream
        count = int(np.sum(mask))
        passed = int(np.sum(ds.passed_mask[mask]))

        sub_scores = ds.final_score[mask]
        top_local_idx = int(np.argmax(sub_scores))
        top_global_idx = np.where(mask)[0][top_local_idx]

        results.append(
            {
                "stream": str(stream),
                "count": count,
                "avg_ia1": float(np.mean(ds.ia1[mask])),
                "avg_ia2": float(np.mean(ds.ia2[mask])),
                "avg_total_ia": float(np.mean(ds.total_ia[mask])),
                "avg_see": float(np.mean(ds.see[mask])),
                "avg_final": float(np.mean(sub_scores)),
                "std_final": float(np.std(sub_scores)),
                "pass_count": passed,
                "fail_count": count - passed,
                "pass_rate": (passed / count) * 100.0,
                "topper_usn": str(ds.usns[top_global_idx]),
                "topper_name": str(ds.names[top_global_idx]),
                "topper_score": float(ds.final_score[top_global_idx]),
            }
        )
    results.sort(key=lambda r: r["avg_final"], reverse=True)
    return results


def get_toppers(ds: StudentDataset, top_n: int = 10) -> np.ndarray:
    """Return indices of top N students by Final Score (tie-broken by SEE, Total IA)."""
    order = np.lexsort((-ds.total_ia, -ds.see, -ds.final_score))
    return order[:top_n]


def get_most_improved(ds: StudentDataset, top_n: int = 10) -> np.ndarray:
    """Return indices of top N students with the largest IA 1 -> IA 2 improvement."""
    return np.argsort(-ds.ia_diff)[:top_n]


def get_failed_indices(ds: StudentDataset) -> np.ndarray:
    """Return indices of all failed students sorted by Final Score ascending."""
    failed_idx = np.where(~ds.passed_mask)[0]
    return failed_idx[np.argsort(ds.final_score[failed_idx])]


def get_at_risk_indices(ds: StudentDataset, margin: float = 5.0) -> np.ndarray:
    """Return passing students who are within `margin` marks of any failing threshold."""
    borderline = ds.passed_mask & (
        (ds.total_ia < MIN_TOTAL_IA + margin)
        | (ds.see < MIN_SEE + margin)
        | (ds.final_score < MIN_FINAL + margin)
    )
    idx = np.where(borderline)[0]
    return idx[np.argsort(ds.final_score[idx])]


def get_outliers(ds: StudentDataset, z_threshold: float = 2.5) -> dict[str, np.ndarray]:
    """Detect statistical outliers using Z-scores and IA-vs-SEE divergence."""
    high_outliers = np.where(ds.z_scores >= z_threshold)[0]
    low_outliers = np.where(ds.z_scores <= -z_threshold)[0]

    # Divergence: normalized IA vs normalized SEE gap >= 30 percentage points
    ia_pct = (ds.total_ia / 50.0) * 100.0
    see_pct = ds.see
    gap = ia_pct - see_pct
    high_ia_low_see = np.where(gap >= 30.0)[0]
    low_ia_high_see = np.where(gap <= -30.0)[0]

    return {
        "high_z": high_outliers[np.argsort(-ds.z_scores[high_outliers])],
        "low_z": low_outliers[np.argsort(ds.z_scores[low_outliers])],
        "high_ia_low_see": high_ia_low_see[np.argsort(-gap[high_ia_low_see])],
        "low_ia_high_see": low_ia_high_see[np.argsort(gap[low_ia_high_see])],
    }


def correlation_matrix(ds: StudentDataset) -> tuple[list[str], np.ndarray]:
    """Compute Pearson correlation matrix across exam components."""
    labels = ["IA 1", "IA 2", "Total IA", "SEE", "Final"]
    matrix = np.vstack([ds.ia1, ds.ia2, ds.total_ia, ds.see, ds.final_score])
    return labels, np.corrcoef(matrix)


def grade_distribution(ds: StudentDataset) -> list[tuple[str, int, float]]:
    """Count students per letter grade in descending grade order."""
    grade_order = ["O", "A+", "A", "B+", "B", "C", "P", "F"]
    unique, counts = np.unique(ds.grades, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    total = len(ds)
    return [
        (g, count_map.get(g, 0), (count_map.get(g, 0) / total) * 100.0)
        for g in grade_order
    ]


def find_student(ds: StudentDataset, query: str) -> np.ndarray:
    """Search students by exact USN or partial case-insensitive name/USN."""
    q = query.strip().upper()
    exact_usn = np.where(np.char.upper(ds.usns) == q)[0]
    if len(exact_usn) > 0:
        return exact_usn
    usn_match = np.char.find(np.char.upper(ds.usns), q) >= 0
    name_match = np.char.find(np.char.upper(ds.names), q) >= 0
    return np.where(usn_match | name_match)[0]


def what_if_see_needed(total_ia: float, target_final: float) -> dict[str, float | bool | str]:
    """Calculate SEE marks (out of 100) needed to reach a target Final Score."""
    # Final = Total IA + SEE / 2  =>  SEE = (Target Final - Total IA) * 2
    raw_see_needed = (target_final - total_ia) * 2.0
    see_needed = max(raw_see_needed, MIN_SEE)
    achievable = (total_ia >= MIN_TOTAL_IA) and (see_needed <= 100.0)
    if total_ia < MIN_TOTAL_IA:
        note = f"Ineligible: Total IA ({total_ia:.1f}) is below minimum {MIN_TOTAL_IA:.0f}."
    elif see_needed > 100.0:
        max_possible = total_ia + 50.0
        note = f"Not achievable (max possible Final Score is {max_possible:.1f})."
    else:
        note = f"Achievable! Needs {see_needed:.1f}/100 in SEE."
    return {
        "total_ia": total_ia,
        "target_final": target_final,
        "see_needed": round(see_needed, 1),
        "achievable": achievable,
        "note": note,
    }


def export_enriched_csv(ds: StudentDataset, output_path: str | Path) -> Path:
    """Export all students with computed Final Score, Grade, Ranks, Percentile, Z-Score."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Sl No",
        "Name",
        "USN",
        "Stream",
        "IA 1 Marks",
        "IA 2 Marks",
        "Total IA",
        "SEE Marks",
        "SEE Reduced (50)",
        "Final Score (100)",
        "IA Improvement",
        "Grade",
        "Status",
        "Overall Rank",
        "Stream Rank",
        "Percentile",
        "Z-Score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(len(ds)):
            writer.writerow(
                [
                    int(ds.sl_no[i]),
                    str(ds.names[i]),
                    str(ds.usns[i]),
                    str(ds.streams[i]),
                    int(ds.ia1[i]),
                    int(ds.ia2[i]),
                    float(ds.total_ia[i]),
                    int(ds.see[i]),
                    float(ds.see_reduced[i]),
                    float(ds.final_score[i]),
                    int(ds.ia_diff[i]),
                    str(ds.grades[i]),
                    str(ds.fail_reason[i]),
                    int(ds.overall_ranks[i]),
                    int(ds.stream_ranks[i]),
                    float(ds.percentiles[i]),
                    round(float(ds.z_scores[i]), 3),
                ]
            )
    return path
