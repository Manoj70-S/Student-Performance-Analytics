"""Generate a realistic 1,000-student dataset segregated by engineering stream."""

import csv
from pathlib import Path
import numpy as np

# Configuration
SEED = 42
STUDENTS_PER_STREAM = 200
STREAMS = {
    "CS": {"mean_ability": 0.72, "std": 0.14},
    "IS": {"mean_ability": 0.70, "std": 0.14},
    "EC": {"mean_ability": 0.65, "std": 0.16},
    "ME": {"mean_ability": 0.62, "std": 0.16},
    "AI": {"mean_ability": 0.73, "std": 0.13},
}

FIRST_NAMES = [
    "Aarav", "Aadhya", "Aditya", "Akash", "Ananya", "Anirudh", "Anjali", "Arjun",
    "Aryan", "Bhavana", "Chaitanya", "Darshan", "Deepak", "Deepika", "Devika", "Dhruv",
    "Divya", "Gaurav", "Harsha", "Harshitha", "Ishaan", "Jhanvi", "Karthik", "Kavya",
    "Keerthi", "Kiran", "Kriti", "Lakshmi", "Madhav", "Manish", "Meghana", "Nandini",
    "Naveen", "Neha", "Nikhil", "Nisha", "Nithin", "Pallavi", "Pavan", "Pooja",
    "Pranav", "Prashanth", "Preethi", "Priya", "Rahul", "Rakshitha", "Ramesh", "Rashmi",
    "Rohan", "Rohit", "Sahana", "Sai", "Samarth", "Sandeep", "Sanjana", "Shreya",
    "Shubham", "Siddharth", "Sindhu", "Sneha", "Srinivas", "Suhas", "Sumanth", "Suraj",
    "Swathi", "Tanvi", "Tarun", "Tejas", "Uday", "Vaishnavi", "Varun", "Vedanth",
    "Vidya", "Vignesh", "Vikas", "Vinay", "Vishal", "Yash", "Yashaswini", "Zoya",
]

LAST_NAMES = [
    "Acharya", "Agarwal", "Bhat", "Chauhan", "Desai", "Deshpande", "Gowda", "Gupta",
    "Hegde", "Iyer", "Jain", "Joshi", "Kamat", "Kulkarni", "Kumar", "Menon",
    "Mishra", "Murthy", "Naidu", "Nair", "Nayak", "Pai", "Patel", "Patil",
    "Prabhu", "Rao", "Reddy", "Sastry", "Sen", "Shah", "Sharma", "Shenoy",
    "Shetty", "Shinde", "Singh", "Sinha", "Subramanian", "Swamy", "Tiwari", "Verma",
    "Yadav", "Raju", "Prasad", "Chandra", "Ranganath", "Srinivasan", "Venkatesh", "Raj",
]

HEADERS = ["Sl No", "Name", "USN", "Stream", "IA 1 Marks", "IA 2 Marks", "Total IA", "SEE Marks"]


def generate_unique_names(rng: np.random.Generator, count: int) -> list[str]:
    """Generate unique realistic full names."""
    all_pairs = [f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES]
    indices = rng.choice(len(all_pairs), size=count, replace=False)
    return [all_pairs[i] for i in indices]


def generate_stream_marks(
    rng: np.random.Generator, count: int, mean_ability: float, std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated IA 1 (0-50), IA 2 (0-50), Total IA (reduced 100->50), and SEE (0-100)."""
    ability = np.clip(rng.normal(mean_ability, std, size=count), 0.15, 0.99)

    ia1 = np.rint(np.clip(ability * 50 + rng.normal(0, 4.4, size=count), 0, 50)).astype(int)
    ia2 = np.rint(np.clip(ability * 50 + rng.normal(1.0, 4.4, size=count), 0, 50)).astype(int)
    total_ia = np.round((ia1 + ia2) / 2.0, 1)
    see = np.rint(np.clip(ability * 100 + rng.normal(-2.0, 9.5, size=count), 0, 100)).astype(int)

    return ia1, ia2, total_ia, see


def write_csv(path: Path, rows: list[list]) -> None:
    """Write rows to a CSV file with standard headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    total_students = STUDENTS_PER_STREAM * len(STREAMS)
    names = generate_unique_names(rng, total_students)

    base_dir = Path(__file__).resolve().parent / "data"
    streams_dir = base_dir / "streams"

    all_rows: list[list] = []
    name_idx = 0
    global_sl = 1

    for stream_code, cfg in STREAMS.items():
        ia1, ia2, total_ia, see = generate_stream_marks(
            rng, STUDENTS_PER_STREAM, cfg["mean_ability"], cfg["std"]
        )
        stream_rows: list[list] = []

        for i in range(STUDENTS_PER_STREAM):
            roll_num = i + 1
            usn = f"1DT24{stream_code}{roll_num:03d}"
            student_name = names[name_idx]
            name_idx += 1

            row_stream = [
                roll_num,
                student_name,
                usn,
                stream_code,
                int(ia1[i]),
                int(ia2[i]),
                float(total_ia[i]),
                int(see[i]),
            ]
            row_master = [
                global_sl,
                student_name,
                usn,
                stream_code,
                int(ia1[i]),
                int(ia2[i]),
                float(total_ia[i]),
                int(see[i]),
            ]
            stream_rows.append(row_stream)
            all_rows.append(row_master)
            global_sl += 1

        stream_file = streams_dir / f"{stream_code}_students.csv"
        write_csv(stream_file, stream_rows)
        print(
            f"Created {stream_file.name}: {len(stream_rows)} students | "
            f"Avg IA1={ia1.mean():.1f}/50, IA2={ia2.mean():.1f}/50, "
            f"Total IA={total_ia.mean():.1f}/50, SEE={see.mean():.1f}/100"
        )

    master_file = base_dir / "students_all.csv"
    write_csv(master_file, all_rows)
    print(f"\nCreated master CSV: {master_file} ({len(all_rows)} total students)")


if __name__ == "__main__":
    main()
