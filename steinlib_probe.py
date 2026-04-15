from pathlib import Path
from dataset_loaders import load_steinlib_graphs

STEINLIB_DIR = Path("steinlib")

graphs = load_steinlib_graphs(STEINLIB_DIR)

total_stp = sum(1 for _ in STEINLIB_DIR.rglob("*.stp"))
loaded    = len(graphs)
failed    = total_stp - loaded

print(f"STP files found : {total_stp}")
print(f"Loaded OK       : {loaded}")
print(f"Parse failures  : {failed}")

if graphs:
    ns = [g.number_of_nodes() for g in graphs.values()]
    ms = [g.number_of_edges() for g in graphs.values()]
    print(f"n  range : {min(ns)} – {max(ns)}")
    print(f"m  range : {min(ms)} – {max(ms)}")

    # Show one sample per subdirectory to confirm subset coverage
    by_subset: dict[str, str] = {}
    for key in graphs:
        subset = key.split("/")[0].split("\\")[0]
        if subset not in by_subset:
            by_subset[subset] = key
    print(f"\nSubsets detected: {len(by_subset)}")
    for subset, example in sorted(by_subset.items()):
        g = graphs[example]
        print(f"  {subset:<12}  example={example}  n={g.number_of_nodes()}  m={g.number_of_edges()}")