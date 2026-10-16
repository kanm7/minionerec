"""Usage diagnostics for RFSQ's fixed implicit scalar-combination spaces."""
import math

try:
    from .sid_metrics import compute_metrics
except ImportError:
    from sid_metrics import compute_metrics


def rfsq_benchmark(indices, levels_per_stage, item_ids=None, **kwargs):
    rows = indices.tolist() if hasattr(indices, "tolist") else indices
    if not rows or any(len(row) != len(levels_per_stage) for row in rows):
        raise ValueError("RFSQ indices must cover every residual stage")
    ids = list(range(len(rows))) if item_ids is None else list(item_ids)
    if len(ids) != len(rows) or len(set(map(str, ids))) != len(ids):
        raise ValueError("RFSQ item IDs must match index rows uniquely")
    sizes = [math.prod(levels) for levels in levels_per_stage]
    invalid = [sum(not 0 <= int(row[s]) < size for row in rows) for s,size in enumerate(sizes)]
    mapping = {str(i): [f"<{chr(97+s)}_{int(code)}>" for s,code in enumerate(row)] for i,row in zip(ids,rows)}
    # Unbounded encoder/residual values may produce invalid flat IDs. Report
    # them, do not clamp or pretend those IDs occupy the finite nominal space.
    report = compute_metrics(mapping, codebook_sizes=[] if any(invalid) else sizes,
                             variant="raw", **kwargs)
    report["quantizer"] = "RFSQ fixed implicit scalar Cartesian product"
    report["learned_codebook_vectors"] = False
    report["levels_per_stage"] = [list(levels) for levels in levels_per_stage]
    report["nominal_stage_capacities"] = sizes
    report["out_of_range_index_counts"] = invalid
    report["all_flat_indices_in_nominal_range"] = not any(invalid)
    for s,layer in enumerate(report["layers"]):
        valid = {int(row[s]) for row in rows if 0 <= int(row[s]) < sizes[s]}
        layer["nominal_capacity"] = sizes[s]
        layer["valid_code_occupancy_rate"] = len(valid)/sizes[s]
        layer["out_of_range_assignments"] = invalid[s]
    if any(invalid):
        report["interpretation"].append("Out-of-range flat IDs found: nominal full/prefix occupancy is unavailable; invalid assignments were not dropped or clamped.")
    report["interpretation"].append("In-range flat IDs do not by themselves prove every scalar coordinate was within its nominal level range.")
    return report
