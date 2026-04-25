import pandas as pd

df = pd.read_csv("results/benchmark_real.csv")

# ── 1. Total unique graph instances ──────────────────────────────────────────
unique_instances = df[["family", "instance"]].drop_duplicates()
total_unique = len(unique_instances)

print("=" * 60)
print("UNIQUE GRAPH INSTANCES TALLY")
print("=" * 60)
print(f"\nTotal unique instances (across all families): {total_unique}")

# ── 2. Unique instances by family ─────────────────────────────────────────────
by_family = (
    df[["family", "instance"]]
    .drop_duplicates()
    .groupby("family")
    .size()
    .rename("unique_instances")
    .sort_values(ascending=False)
)
print("\nUnique instances per family:")
print(by_family.to_string())

# ── 3. Execution status breakdown (rows, not instances) ──────────────────────
status_counts = df["status"].value_counts()
print("\n" + "=" * 60)
print("EXECUTION STATUS BREAKDOWN  (row-level counts)")
print("=" * 60)
print(status_counts.to_string())

# Unique instances that saw at least one skipped row
skipped_instances = (
    df[df["status"] == "skipped_greedy_n_gt_500"][["family", "instance"]]
    .drop_duplicates()
)
print(f"\nUnique instances with at least one skipped row: {len(skipped_instances)}")

# ── 4. Algorithm-level unique instance counts ─────────────────────────────────
bs_ok = (
    df[(df["algo"] == "BS") & (df["status"] == "ok")][["family", "instance"]]
    .drop_duplicates()
)
greedy_ok = (
    df[(df["algo"] == "Greedy") & (df["status"] == "ok")][["family", "instance"]]
    .drop_duplicates()
)

print("\n" + "=" * 60)
print("ALGORITHM-LEVEL UNIQUE INSTANCE COUNTS  (status == ok)")
print("=" * 60)
print(f"  Baswana-Sen (BS)  successfully evaluated: {len(bs_ok):>5}")
print(f"  Greedy            successfully evaluated: {len(greedy_ok):>5}")

both_ok = pd.merge(bs_ok, greedy_ok, on=["family", "instance"])
print(f"  Both algorithms   successfully evaluated: {len(both_ok):>5}")

# ── 5. Final verified count ───────────────────────────────────────────────────
if "verified" in df.columns:
    verified_instances = (
        df[df["verified"] == True][["family", "instance"]]
        .drop_duplicates()
    )
    final_valid = len(verified_instances)
else:
    # Fall back: instances where every row is ok
    all_ok = df.groupby(["family", "instance"])["status"].apply(
        lambda s: (s == "ok").all()
    )
    final_valid = all_ok.sum()

print("\n" + "=" * 60)
print(f"FINAL VERIFIED COUNT OF VALID INSTANCES: {final_valid}")
print("=" * 60)
