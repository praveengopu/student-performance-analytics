from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    df["performance_category"] = pd.cut(
        df["final_marks"],
        bins=[0, 40, 60, 75, 100],
        labels=["Needs Support", "Average", "Good", "Excellent"],
        include_lowest=True
    )

    summary_by_study_hours = (
        df.groupby("study_hours_group", observed=False)["final_marks"]
        .agg(["count", "mean", "min", "max"])
        .round(2)
        .reset_index()
    )
    summary_by_study_hours.to_csv(
        RESULTS_DIR / "summary_by_study_hours.csv",
        index=False
    )

    category_counts = (
        df["performance_category"]
        .value_counts()
        .rename_axis("performance_category")
        .reset_index(name="student_count")
    )
    category_counts.to_csv(
        RESULTS_DIR / "performance_category_counts.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(df["attendance_percentage"], df["final_marks"], alpha=0.7)
    plt.title("Attendance Percentage vs Final Marks")
    plt.xlabel("Attendance Percentage")
    plt.ylabel("Final Marks")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "attendance_vs_final_marks.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    df.groupby("study_hours_group", observed=False)["final_marks"].mean().plot(kind="bar")
    plt.title("Average Final Marks by Study Hours Group")
    plt.xlabel("Study Hours Group")
    plt.ylabel("Average Final Marks")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "average_marks_by_study_hours.png", dpi=150)
    plt.close()

    print("\nSummary by study hours:")
    print(summary_by_study_hours.to_string(index=False))
    print("\nSaved analysis outputs in the results folder.")

if __name__ == "__main__":
    main()
