#!/usr/bin/env python3
"""Diagnose SID capacity, exclusivity, concentration and label purity."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


def normalized_entropy(counts):
    total = sum(counts.values())
    if not total or len(counts) <= 1: return 0.0
    return -sum((n/total)*math.log(n/total) for n in counts.values() if n) / math.log(len(counts))


def extract_label(item, field, level="leaf"):
    value = item
    for key in field.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    if isinstance(value, list):
        if value and isinstance(value[0], list): value = value[0]  # first taxonomy path
        values = [str(v).strip() for v in value if v is not None and str(v).strip()]
        value = (values[-1] if level == "leaf" else values[0]) if values else None
    if value is None or str(value).strip().casefold() in ("", "unknown", "none", "null", "n/a"): return None
    return str(value).strip()


def load_items(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f: return {str(r["item_id"]): r for r in map(json.loads, f)}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict): raise ValueError("Item JSON must map item IDs to metadata")
    return {str(k):v for k,v in values.items()}


def purity_metrics(groups, labels):
    clusters = {key:Counter(labels[i] for i in ids if labels.get(i) is not None) for key,ids in groups.items()}
    clusters = {k:c for k,c in clusters.items() if c}
    labeled = sum(sum(c.values()) for c in clusters.values())
    correct = sum(max(c.values()) for c in clusters.values())
    # Use original cluster size: an unlabeled companion must not turn a cluster into a singleton.
    non_singletons = {k:c for k,c in clusters.items() if len(groups[k]) > 1}
    non_total = sum(sum(c.values()) for c in non_singletons.values())
    return {
        "labeled_items":labeled, "labeled_item_coverage":labeled/max(sum(map(len,groups.values())),1),
        "distinct_labels":len(set(labels.values())-{None}), "labeled_clusters":len(clusters),
        "item_weighted_purity":correct/labeled if labeled else None,
        "macro_cluster_purity":sum(max(c.values())/sum(c.values()) for c in clusters.values())/len(clusters) if clusters else None,
        "non_singleton_cluster_item_weighted_purity":sum(max(c.values()) for c in non_singletons.values())/non_total if non_total else None,
        "non_singleton_labeled_items":non_total,
        "labeled_item_share_in_singleton_clusters":sum(sum(c.values()) for k,c in clusters.items() if len(groups[k])==1)/labeled if labeled else None,
    }


def cluster_metrics(groups, capacity=None, popularity=None, labels=None, hot_percent=1., hot_multiplier=5.):
    counts = Counter({k:len(v) for k,v in groups.items()})
    total = sum(counts.values()); used = len(counts)
    singleton = sum(n==1 for n in counts.values())
    top_n = max(1, math.ceil(used*hot_percent/100))
    sorted_sizes = sorted(counts.values(),reverse=True)
    probabilities = [n/total for n in counts.values()]
    entropy = -sum(p*math.log(p) for p in probabilities)
    result = {
        "used_codes":used, "capacity":capacity, "occupancy_rate":used/capacity if capacity else None,
        "unused_codes":capacity-used if capacity else None,
        "singleton_codes":singleton, "singleton_rate":singleton/used,
        "exclusive_item_rate":singleton/total,
        "collision_item_rate":sum(n for n in counts.values() if n>1)/total,
        "excess_assignment_collision_rate":(total-used)/total,
        "normalized_entropy":normalized_entropy(counts),
        "capacity_normalized_entropy":entropy/math.log(capacity) if capacity and capacity>1 else (0. if capacity==1 else None),
        "effective_codes":math.exp(entropy),
        "effective_capacity_rate":math.exp(entropy)/capacity if capacity else None,
        "max_cluster_size":max(sorted_sizes), "max_code_item_share":max(sorted_sizes)/total,
        "mean_cluster_size":total/used, "top_1pct_item_share":sum(sorted_sizes[:max(1,math.ceil(used*.01))])/total,
        "largest_clusters": {"percent":hot_percent,"cluster_count":top_n,"item_share":sum(sorted_sizes[:top_n])/total},
    }
    if popularity is not None:
        masses = {key:sum(popularity.get(i,0.) for i in ids) for key,ids in groups.items()}
        mass_total = sum(masses.values())
        # Stable sorting for zero/tied popularity; never rank using test interactions.
        ranked = sorted(groups,key=lambda k:(-masses[k],str(k)))[:top_n]
        result["hot_clusters"] = {
            "ranked_by":"summed training interaction counts", "percent":hot_percent,
            "cluster_count":top_n, "cluster_share":top_n/used,
            "item_share":sum(len(groups[k]) for k in ranked)/total if mass_total else None,
            "popularity_share":sum(masses[k] for k in ranked)/mass_total if mass_total else None,
            "total_popularity":mass_total,
        }
        threshold = hot_multiplier * mass_total / used
        hot = [k for k in groups if mass_total and masses[k] >= threshold]
        result["hot_clusters_above_mean"] = {
            "mean_multiplier":hot_multiplier, "interaction_threshold":threshold if mass_total else None,
            "cluster_count":len(hot) if mass_total else None,
            "cluster_rate":len(hot)/used if mass_total else None,
            "item_share":sum(len(groups[k]) for k in hot)/total if mass_total else None,
            "popularity_share":sum(masses[k] for k in hot)/mass_total if mass_total else None,
        }
    if labels is not None: result["purity"] = purity_metrics(groups,labels)
    return result


def validate_indices(indices, sizes):
    if not isinstance(indices,dict) or not indices: raise ValueError("Index must be a nonempty item-ID mapping")
    rows = list(indices.values())
    if not isinstance(rows[0],list) or not rows[0]: raise ValueError("SID must be a nonempty token list")
    depth = len(rows[0])
    if any(not isinstance(r,list) or len(r)!=depth or any(not isinstance(t,str) for t in r) for r in rows):
        raise ValueError("SID depth must be uniform and tokens must be strings")
    if sizes and (len(sizes)!=depth or any(s<=0 for s in sizes)):
        raise ValueError("Provide one positive capacity per SID layer")
    for level in range(depth):
        if sizes and len({r[level] for r in rows})>sizes[level]: raise ValueError("Used codes exceed layer capacity")
        for row in rows:
            match = re.fullmatch(r"<[a-z]_(\d+)>",row[level])
            if sizes and match and int(match[1])>=sizes[level]: raise ValueError("SID token outside declared codebook capacity")
    return depth


def compute_metrics(indices, codebook_sizes=(), item_meta=None, category_field="categories", category_level="leaf",
                    popularity=None, hot_percent=1., variant="unknown", hot_multiplier=5.):
    depth = validate_indices(indices,codebook_sizes)
    if not 0<hot_percent<=100: raise ValueError("Hot cluster percent must be in (0,100]")
    if not math.isfinite(hot_multiplier) or hot_multiplier<=0:raise ValueError("Hot multiplier must be positive and finite")
    if popularity is not None and any(not math.isfinite(v) or v<0 for v in popularity.values()):
        raise ValueError("Popularity values must be finite and nonnegative")
    labels = {str(i):extract_label((item_meta or {}).get(str(i),{}),category_field,category_level) for i in indices} if item_meta is not None else None
    def group(kind,level):
        groups = defaultdict(list)
        for iid,row in indices.items(): groups[row[level-1] if kind=="layer" else tuple(row[:level])].append(str(iid))
        return groups
    full = group("prefix",depth)
    full_capacity = math.prod(codebook_sizes) if codebook_sizes else None
    full_metrics = cluster_metrics(full,full_capacity,popularity,labels,hot_percent,hot_multiplier)
    result = {"items":len(indices),"depth":depth,"sid_variant":variant,
              "unique_sid_rate":len(full)/len(indices),"collision_rate":(len(indices)-len(full))/len(indices),
              "full_sid":full_metrics,"layers":[],"prefix":[],
              "definitions":{
                  "singleton_rate":"codes assigned to exactly one catalog item / used codes",
                  "exclusive_item_rate":"items whose code/cluster is exclusive / all catalog items",
                  "occupancy_rate":"used codes / declared capacity (prefix capacity is the product of layer capacities)",
                  "collision_rate":"excess assignments (items - distinct full SIDs) / items",
                  "collision_item_rate":"all items in non-singleton clusters / items",
                  "hot_clusters":"top ceil(used clusters * percent / 100), ranked by summed training interactions; ties by code",
                  "hot_clusters_above_mean":"clusters with training interaction mass >= multiplier * mean mass of used clusters",
                  "purity":"sum of majority-label item counts per cluster / labeled items; unlabeled items excluded and coverage reported",
                  "normalized_entropy":"entropy / log(used codes); capacity_normalized_entropy uses log(declared capacity)",
              },
              "interpretation":["Singleton clusters have purity 1 by construction; inspect non-singleton purity and coverage.",
                                "Collision repair may increase exclusivity/purity mechanically; compare raw and repaired SIDs.",
                                "Prefix capacity grows multiplicatively; low full-SID occupancy can be expected even with good layer occupancy.",
                                "These diagnostics do not establish recommendation Recall/NDCG or image-background invariance."]}
    for level in range(1,depth+1):
        layer=cluster_metrics(group("layer",level),codebook_sizes[level-1] if codebook_sizes else None,popularity,labels,hot_percent,hot_multiplier)
        layer["layer"]=level;result["layers"].append(layer)
        prefix=cluster_metrics(group("prefix",level),math.prod(codebook_sizes[:level]) if codebook_sizes else None,popularity,labels,hot_percent,hot_multiplier)
        prefix["depth"]=level;result["prefix"].append(prefix)
    if labels is not None:
        result["category_config"]={"field":category_field,"level":category_level,"multiple_paths":"first path"}
        result["full_sid_category_purity"]=full_metrics["purity"]["item_weighted_purity"]
    if popularity is not None:
        result["popularity_coverage"]={"items_with_recorded_popularity":sum(str(i) in popularity for i in indices),
                                       "items_with_positive_popularity":sum(popularity.get(str(i),0)>0 for i in indices)}
    return result


def prefix_metrics(rows, depth):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[tuple(row[:depth])].append(str(i))
    return dict(cluster_metrics(groups),depth=depth)


def category_purity(indices,item_meta,category_field):
    return compute_metrics(indices,item_meta=item_meta,category_field=category_field)["full_sid_category_purity"]


def load_popularity(path, split="train"):
    if Path(path).suffix != ".jsonl": raise ValueError("Interactions must be raw interaction JSONL, not overlapping sequence histories")
    counts=Counter()
    with Path(path).open(encoding="utf-8") as f:
        for row in map(json.loads,f):
            if "split" not in row:raise ValueError("Interaction records need split labels to avoid test popularity leakage")
            if row["split"]==split:counts[str(row["item_id"])]+=1
    return dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index_file",required=True)
    p.add_argument("--codebook_sizes",type=int,nargs="*",default=[])
    p.add_argument("--item_file",default="")
    p.add_argument("--category_field",default="categories")
    p.add_argument("--category_level",choices=("leaf","root"),default="leaf")
    p.add_argument("--interaction_file",help="Raw interaction JSONL with split labels; use training records only")
    p.add_argument("--hot_cluster_percent",type=float,default=1.)
    p.add_argument("--hot_cluster_multiplier",type=float,default=5.)
    p.add_argument("--sid_variant",choices=("raw","repaired","unknown"),default="unknown")
    p.add_argument("--raw_index_file",help="Optional raw SID mapping to compare with the main repaired mapping")
    p.add_argument("--output_file",default="")
    a=p.parse_args()
    if a.output_file:
        inputs=[a.index_file,a.raw_index_file,a.item_file,a.interaction_file]
        if any(Path(a.output_file).resolve()==Path(source).resolve() for source in inputs if source):
            p.error("Metric output must not overwrite an input file")
    indices=json.loads(Path(a.index_file).read_text(encoding="utf-8"))
    meta=load_items(a.item_file) if a.item_file else None
    pop=load_popularity(a.interaction_file) if a.interaction_file else None
    kwargs=dict(codebook_sizes=a.codebook_sizes,item_meta=meta,category_field=a.category_field,
                category_level=a.category_level,popularity=pop,hot_percent=a.hot_cluster_percent,
                hot_multiplier=a.hot_cluster_multiplier)
    result=compute_metrics(indices,variant=a.sid_variant,**kwargs)
    if a.raw_index_file:
        raw=json.loads(Path(a.raw_index_file).read_text(encoding="utf-8"))
        if set(raw)!=set(indices):raise ValueError("Raw/repaired comparisons must cover identical item IDs")
        if validate_indices(raw,a.codebook_sizes)!=result["depth"]:raise ValueError("Raw/repaired depths differ")
        result["raw_comparison"]=compute_metrics(raw,variant="raw",**kwargs)
    text=json.dumps(result,ensure_ascii=False,indent=2);print(text)
    if a.output_file:
        target=Path(a.output_file);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding="utf-8")


if __name__=="__main__":main()
