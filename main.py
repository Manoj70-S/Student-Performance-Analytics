"""Interactive CLI & Report Runner for Student Performance Analytics (NumPy)."""

import argparse
from pathlib import Path
import numpy as np

from analytics import (
    StudentDataset,
    correlation_matrix,
    exam_summary_stats,
    export_enriched_csv,
    find_student,
    get_at_risk_indices,
    get_failed_indices,
    get_most_improved,
    get_outliers,
    get_toppers,
    grade_distribution,
    load_dataset,
    stream_comparison,
    what_if_see_needed,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "students_all.csv"
DEFAULT_EXPORT = BASE_DIR / "output" / "student_results_report.csv"


def print_header(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}")


def show_overview(ds: StudentDataset, label: str = "ALL STREAMS") -> None:
    print_header(f"PERFORMANCE OVERVIEW — {label} (N = {len(ds)})")
    passed = int(np.sum(ds.passed_mask))
    failed = len(ds) - passed
    print(
        f"  Passed: {passed} ({passed / len(ds) * 100:.1f}%)  |  "
        f"Failed: {failed} ({failed / len(ds) * 100:.1f}%)"
    )

    stats = exam_summary_stats(ds)
    print(
        f"\n  {'Metric':<18} {'Mean':>7} {'Median':>7} {'Std':>7} "
        f"{'Min':>6} {'Max':>6} {'Q25':>6} {'Q75':>6} {'P90':>6}"
    )
    print("  " + "-" * 73)
    for name, s in stats.items():
        print(
            f"  {name:<18} {s['mean']:>7.2f} {s['median']:>7.2f} {s['std']:>7.2f} "
            f"{s['min']:>6.1f} {s['max']:>6.1f} {s['q25']:>6.1f} {s['q75']:>6.1f} {s['p90']:>6.1f}"
        )


def show_stream_comparison(ds: StudentDataset) -> None:
    print_header("STREAM-WISE COMPARISON (RANKED BY AVG FINAL SCORE)")
    rows = stream_comparison(ds)
    print(
        f"  {'Stream':<6} {'N':>4} {'IA1':>6} {'IA2':>6} {'TotIA':>6} {'SEE':>6} "
        f"{'Final':>6} {'Std':>5} {'Pass%':>7}  {'Topper (Score)'}"
    )
    print("  " + "-" * 76)
    for r in rows:
        print(
            f"  {r['stream']:<6} {r['count']:>4} {r['avg_ia1']:>6.1f} {r['avg_ia2']:>6.1f} "
            f"{r['avg_total_ia']:>6.1f} {r['avg_see']:>6.1f} {r['avg_final']:>6.1f} "
            f"{r['std_final']:>5.1f} {r['pass_rate']:>6.1f}%  "
            f"{r['topper_name']} ({r['topper_score']:.1f})"
        )


def show_toppers(ds: StudentDataset, top_n: int = 10) -> None:
    print_header(f"TOP {top_n} OVERALL TOPPERS")
    idx = get_toppers(ds, top_n=top_n)
    print(
        f"  {'Rank':<5} {'USN':<12} {'Name':<22} {'Str':<4} "
        f"{'IA1':>4} {'IA2':>4} {'TotIA':>6} {'SEE':>4} {'Final':>6} {'Grd':>4}"
    )
    print("  " + "-" * 75)
    for i in idx:
        print(
            f"  {ds.overall_ranks[i]:<5} {ds.usns[i]:<12} {ds.names[i]:<22} {ds.streams[i]:<4} "
            f"{ds.ia1[i]:>4.0f} {ds.ia2[i]:>4.0f} {ds.total_ia[i]:>6.1f} {ds.see[i]:>4.0f} "
            f"{ds.final_score[i]:>6.1f} {ds.grades[i]:>4}"
        )

    print_header(f"TOP {min(5, len(ds))} MOST IMPROVED (IA 1 -> IA 2)")
    imp_idx = get_most_improved(ds, top_n=min(5, len(ds)))
    print(f"  {'USN':<12} {'Name':<22} {'Str':<4} {'IA1':>5} {'IA2':>5} {'Gain':>6} {'Final':>7}")
    print("  " + "-" * 65)
    for i in imp_idx:
        print(
            f"  {ds.usns[i]:<12} {ds.names[i]:<22} {ds.streams[i]:<4} "
            f"{ds.ia1[i]:>5.0f} {ds.ia2[i]:>5.0f} {ds.ia_diff[i]:>+6.0f} {ds.final_score[i]:>7.1f}"
        )


def show_failed_and_outliers(ds: StudentDataset, limit: int = 15) -> None:
    failed_idx = get_failed_indices(ds)
    at_risk_idx = get_at_risk_indices(ds, margin=5.0)
    outliers = get_outliers(ds, z_threshold=2.5)

    print_header(f"FAILED & AT-RISK STUDENTS")
    print(f"  Total Failed Students : {len(failed_idx)}")
    if len(failed_idx) > 0:
        reasons, counts = np.unique(ds.fail_reason[failed_idx], return_counts=True)
        breakdown = ", ".join(f"{r}: {c}" for r, c in zip(reasons, counts))
        print(f"  Failure Breakdown     : {breakdown}")
        print(f"\n  Lowest {min(limit, len(failed_idx))} Failed Students:")
        print(
            f"  {'USN':<12} {'Name':<22} {'Str':<4} {'TotIA':>6} {'SEE':>5} {'Final':>6}  {'Reason'}"
        )
        print("  " + "-" * 72)
        for i in failed_idx[:limit]:
            print(
                f"  {ds.usns[i]:<12} {ds.names[i]:<22} {ds.streams[i]:<4} "
                f"{ds.total_ia[i]:>6.1f} {ds.see[i]:>5.0f} {ds.final_score[i]:>6.1f}  "
                f"{ds.fail_reason[i]}"
            )

    print(f"\n  Borderline / At-Risk Passing Students (within 5 marks of failing): {len(at_risk_idx)}")
    print(
        f"  Statistical Outliers (|Z| >= 2.5): "
        f"{len(outliers['high_z'])} high, {len(outliers['low_z'])} low"
    )
    print(
        f"  IA vs SEE Divergence (>= 30% gap): "
        f"{len(outliers['high_ia_low_see'])} High-IA/Low-SEE, "
        f"{len(outliers['low_ia_high_see'])} Low-IA/High-SEE"
    )


def show_correlations_and_grades(ds: StudentDataset) -> None:
    print_header("CORRELATION MATRIX (PEARSON r)")
    labels, corr = correlation_matrix(ds)
    header = "  " + f"{'':<10}" + "".join(f"{lbl:>10}" for lbl in labels)
    print(header)
    print("  " + "-" * (10 + 10 * len(labels)))
    for lbl, row in zip(labels, corr):
        vals = "".join(f"{v:>10.3f}" for v in row)
        print(f"  {lbl:<10}{vals}")

    print_header("GRADE DISTRIBUTION")
    for grade, count, pct in grade_distribution(ds):
        bar = "#" * int(round(pct / 2))
        print(f"  {grade:<3} | {count:>4} ({pct:>5.1f}%) | {bar}")


def show_student_card(ds: StudentDataset, idx: int) -> None:
    stream = ds.streams[idx]
    stream_mask = ds.streams == stream
    stream_avg = float(np.mean(ds.final_score[stream_mask]))
    overall_avg = float(np.mean(ds.final_score))

    print_header(f"STUDENT REPORT CARD — {ds.names[idx]} ({ds.usns[idx]})")
    print(f"  Stream          : {stream}")
    print(f"  IA 1 / IA 2     : {ds.ia1[idx]:.0f}/50  |  {ds.ia2[idx]:.0f}/50  (Change: {ds.ia_diff[idx]:+.0f})")
    print(f"  Total IA        : {ds.total_ia[idx]:.1f} / 50")
    print(f"  SEE Marks       : {ds.see[idx]:.0f} / 100  (Reduced: {ds.see_reduced[idx]:.1f} / 50)")
    print(f"  Final Score     : {ds.final_score[idx]:.1f} / 100")
    print(f"  Grade & Status  : {ds.grades[idx]}  ({ds.fail_reason[idx]})")
    print(f"  Overall Rank    : #{ds.overall_ranks[idx]} / {len(ds)}  (Percentile: {ds.percentiles[idx]:.1f}%)")
    print(f"  Stream Rank     : #{ds.stream_ranks[idx]} / {int(np.sum(stream_mask))}")
    print(f"  Z-Score         : {ds.z_scores[idx]:+.2f}")
    print(
        f"  Vs Averages     : {ds.final_score[idx] - stream_avg:+.1f} vs {stream} avg ({stream_avg:.1f})  |  "
        f"{ds.final_score[idx] - overall_avg:+.1f} vs overall avg ({overall_avg:.1f})"
    )


def compare_two_students(ds: StudentDataset, usn1: str, usn2: str) -> None:
    idx1 = find_student(ds, usn1)
    idx2 = find_student(ds, usn2)
    if len(idx1) == 0 or len(idx2) == 0:
        print("  Could not find one or both students. Check the USNs and try again.")
        return
    a, b = int(idx1[0]), int(idx2[0])
    print_header(f"STUDENT COMPARISON: {ds.usns[a]} vs {ds.usns[b]}")
    print(f"  {'Metric':<18} {str(ds.names[a])[:22]:>22} {str(ds.names[b])[:22]:>22} {'Diff (A-B)':>12}")
    print("  " + "-" * 76)
    rows = [
        ("USN", str(ds.usns[a]), str(ds.usns[b]), ""),
        ("Stream", str(ds.streams[a]), str(ds.streams[b]), ""),
        ("IA 1 (/50)", f"{ds.ia1[a]:.0f}", f"{ds.ia1[b]:.0f}", f"{ds.ia1[a] - ds.ia1[b]:+.0f}"),
        ("IA 2 (/50)", f"{ds.ia2[a]:.0f}", f"{ds.ia2[b]:.0f}", f"{ds.ia2[a] - ds.ia2[b]:+.0f}"),
        ("Total IA (/50)", f"{ds.total_ia[a]:.1f}", f"{ds.total_ia[b]:.1f}", f"{ds.total_ia[a] - ds.total_ia[b]:+.1f}"),
        ("SEE (/100)", f"{ds.see[a]:.0f}", f"{ds.see[b]:.0f}", f"{ds.see[a] - ds.see[b]:+.0f}"),
        ("Final (/100)", f"{ds.final_score[a]:.1f}", f"{ds.final_score[b]:.1f}", f"{ds.final_score[a] - ds.final_score[b]:+.1f}"),
        ("Grade", str(ds.grades[a]), str(ds.grades[b]), ""),
        ("Overall Rank", f"#{ds.overall_ranks[a]}", f"#{ds.overall_ranks[b]}", f"{ds.overall_ranks[b] - ds.overall_ranks[a]:+d}"),
        ("Percentile", f"{ds.percentiles[a]:.1f}%", f"{ds.percentiles[b]:.1f}%", f"{ds.percentiles[a] - ds.percentiles[b]:+.1f}%"),
    ]
    for metric, va, vb, diff in rows:
        print(f"  {metric:<18} {va:>22} {vb:>22} {diff:>12}")


def run_full_report(ds: StudentDataset) -> None:
    show_overview(ds)
    show_stream_comparison(ds)
    show_toppers(ds, top_n=10)
    show_failed_and_outliers(ds, limit=10)
    show_correlations_and_grades(ds)
    exported = export_enriched_csv(ds, DEFAULT_EXPORT)
    print(f"\nExported enriched report CSV -> {exported}")


def interactive_menu(ds: StudentDataset) -> None:
    active_ds = ds
    active_label = "ALL STREAMS"

    while True:
        print(f"\n--- Student Performance Analytics [{active_label} | N={len(active_ds)}] ---")
        print("  1. Overview & Stream Comparison")
        print("  2. Toppers & Most Improved (IA 1 -> IA 2)")
        print("  3. Failed, At-Risk & Outlier Analysis")
        print("  4. Correlation Matrix & Grade Distribution")
        print("  5. Lookup Student (by USN or Name)")
        print("  6. Compare Two Students (by USN)")
        print("  7. 'What-If' SEE Target Calculator")
        print("  8. Filter by Stream (CS / IS / EC / ME / AI / ALL)")
        print("  9. Export Enriched Results CSV")
        print("  0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            show_overview(active_ds, active_label)
            if active_label == "ALL STREAMS":
                show_stream_comparison(active_ds)
        elif choice == "2":
            show_toppers(active_ds, top_n=10)
        elif choice == "3":
            show_failed_and_outliers(active_ds, limit=15)
        elif choice == "4":
            show_correlations_and_grades(active_ds)
        elif choice == "5":
            query = input("Enter USN or Name: ").strip()
            matches = find_student(ds, query)
            if len(matches) == 0:
                print("  No matching student found.")
            elif len(matches) == 1:
                show_student_card(ds, int(matches[0]))
            else:
                print(f"  Found {len(matches)} matches (showing first 10):")
                for idx in matches[:10]:
                    print(f"    {ds.usns[idx]} | {ds.names[idx]} ({ds.streams[idx]}) - Final: {ds.final_score[idx]:.1f}")
                pick = input("  Enter exact USN to view full card (or Enter to skip): ").strip()
                if pick:
                    exact = find_student(ds, pick)
                    if len(exact) > 0:
                        show_student_card(ds, int(exact[0]))
        elif choice == "6":
            u1 = input("Enter First USN  (e.g. 1DT24CS001): ").strip()
            u2 = input("Enter Second USN (e.g. 1DT24AI001): ").strip()
            compare_two_students(ds, u1, u2)
        elif choice == "7":
            try:
                total_ia = float(input("Enter student's Total IA (out of 50): "))
                target = float(input("Enter desired Final Score (out of 100): "))
                res = what_if_see_needed(total_ia, target)
                print(f"\n  Result: {res['note']}")
            except ValueError:
                print("  Please enter valid numeric values.")
        elif choice == "8":
            s = input("Enter stream (CS, IS, EC, ME, AI, or ALL): ").strip().upper()
            if s == "ALL":
                active_ds = ds
                active_label = "ALL STREAMS"
            else:
                try:
                    active_ds = ds.filter_stream(s)
                    active_label = f"STREAM: {s}"
                    print(f"  Switched filter to {active_label} ({len(active_ds)} students).")
                except ValueError as e:
                    print(f"  {e}")
        elif choice == "9":
            suffix = "" if active_label == "ALL STREAMS" else f"_{active_label.split(': ')[1]}"
            out_path = BASE_DIR / "output" / f"student_results_report{suffix}.csv"
            exported = export_enriched_csv(active_ds, out_path)
            print(f"  Exported {len(active_ds)} rows to: {exported}")
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("  Invalid choice. Please select 0-9.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy Student Performance Analytics")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to student CSV")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print full analytics report and export enriched CSV without interactive menu",
    )
    parser.add_argument("--stream", type=str, help="Optional stream filter (CS, IS, EC, ME, AI)")
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    if args.stream:
        ds = ds.filter_stream(args.stream)

    if args.report:
        run_full_report(ds)
    else:
        interactive_menu(ds)


if __name__ == "__main__":
    main()
