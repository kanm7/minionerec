"""Resume structured attributes, run mixed embeddings, and export verified SID inputs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import fcntl

import requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--ollama_model", required=True)
    p.add_argument("--ollama_url", default="http://127.0.0.1:11435")
    p.add_argument("--num_ctx", type=int, default=8192)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--mmproj", type=Path, required=True)
    p.add_argument("--llama_server", type=Path, required=True)
    p.add_argument("--port", type=int, default=8912)
    p.add_argument("--matrix_output", type=Path, required=True)
    p.add_argument("--detach", action="store_true", help="Run as a detached background process; record its PID")
    p.add_argument("--keep_awake", action="store_true", help="Keep macOS from idle-sleeping while a detached task runs")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.detach:
        args = [arg for arg in sys.argv[1:] if arg not in ("--detach", "--keep_awake")]
        with (a.output_dir / "pipeline-background.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *args],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, cwd=Path(__file__).resolve().parents[1])
        (a.output_dir / "pipeline.pid").write_text(str(process.pid)+"\n")
        if a.keep_awake and sys.platform == "darwin":
            subprocess.Popen(["caffeinate", "-i", "-w", str(process.pid)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print("Background pipeline PID:", process.pid)
        return
    with a.catalog.open(encoding="utf-8") as f: catalog = {str(x["item_id"]): x for x in map(json.loads, f)}
    attempts = a.output_dir / "attributes-attempts.jsonl"
    report = {"status": "running", "catalog_items": len(catalog), "commands": {}, "timings": {}}
    report_path = a.output_dir / "pipeline.status.json"
    def save(): report_path.write_text(json.dumps(report, indent=2))
    def run(name, command):
        report["commands"][name] = shlex.join(command); save()
        print("RUN", name, shlex.join(command), flush=True)
        with (a.output_dir / (name + ".log")).open("w") as log:
            start = time.perf_counter()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        report["timings"][name] = time.perf_counter()-start; save()
    def latest_records():
        if not attempts.exists(): return {}
        with attempts.open(encoding="utf-8") as f: records = {str(x["item_id"]): x for x in map(json.loads, f)}
        for iid, r in records.items():
            if iid not in catalog or r.get("parent_asin") != catalog[iid]["parent_asin"]:
                raise ValueError(f"Resume file/catalog ID mismatch: {iid}")
        return records
    server = None
    gpu_lock = None
    try:
        report['stage'] = 'waiting_for_gpu'; save()
        gpu_lock = (Path(__file__).resolve().parents[1]/'output/.local-model-gpu.lock').open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        report['stage'] = 'attributes'; save()
        existing = latest_records()
        report["attribute_resumed_successes"] = sum(bool(r.get("summary_text")) and not r.get("error") for r in existing.values())
        attribute_cmd = [sys.executable, "rq/summary/generate_structured_summaries.py", "--item_file", str(a.catalog),
                         "--output_file", str(attempts), "--ollama_model", a.ollama_model,
                         "--ollama_url", a.ollama_url, "--allow_missing_images", "--workers", str(a.workers),
                         "--num_ctx", str(a.num_ctx), "--keep_alive", "-1"]
        for attempt in range(3):
            run(f"attributes-attempt-{attempt+1}", attribute_cmd)
            records = latest_records()
            failures = [iid for iid, item in catalog.items() if item.get("text_usable") is not False and
                        (iid not in records or records[iid].get("error") or not records[iid].get("summary_text"))]
            if not failures: break
            print("Attribute failures to retry", failures, flush=True)
        else: raise ValueError(f"Unresolved attribute failures: {failures}")
        attributes = a.output_dir / "attributes.jsonl"
        with attributes.open("w", encoding="utf-8") as f:
            for iid in sorted(catalog, key=int):
                r = records[iid]
                if catalog[iid].get("text_usable") is not False:
                    assert r["summary_version"] == "product-attributes-v2" and r["summary_text"] != "unknown product"
                    assert r["summary_image_used"] == bool(catalog[iid].get("images"))
                f.write(json.dumps(r, ensure_ascii=False)+"\n")
        report["attribute_successes"] = len(catalog) - sum(i.get("text_usable") is False for i in catalog.values())
        report["unembeddable_items"] = sum(i.get("text_usable") is False for i in catalog.values()); save()
        r = requests.post(a.ollama_url+"/api/generate", json={"model":a.ollama_model,"keep_alive":0,"stream":False},timeout=600)
        r.raise_for_status()
        assert not requests.get(a.ollama_url+"/api/ps",timeout=10).json()["models"], "Ollama model still loaded"
        shard_dir = a.output_dir / ("mixed-"+time.strftime("%Y%m%d-%H%M%S"))
        cmd = [str(a.llama_server), "-m",str(a.model),"--mmproj",str(a.mmproj),"--embedding","--pooling","last",
               "--port",str(a.port),"-ngl","99","-c","8192","-b","2048","-ub","2048","--parallel","1"]
        report["commands"]["embedding-server"] = shlex.join(cmd); save()
        log = (a.output_dir/"embedding-server.log").open("w")
        start = time.perf_counter(); server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        url=f"http://127.0.0.1:{a.port}"
        for _ in range(600):
            if server.poll() is not None: raise RuntimeError("Embedding server exited")
            try:
                if requests.get(url+"/health",timeout=1).status_code==200: break
            except requests.RequestException: pass
            time.sleep(.1)
        else: raise RuntimeError("Embedding server startup timed out")
        report["timings"]["embedding_load"] = time.perf_counter()-start; save()
        run("mixed-embedding",[sys.executable,"rq/text2emb/amazon23_summary_image_embed.py","--catalog",str(a.catalog),
                               "--summary_path",str(attributes),"--output_dir",str(shard_dir),"--model",str(a.model),
                               "--mmproj",str(a.mmproj),"--embedding_mode","mixed","--server_url",url,
                               "--allow_unembeddable","--shard_size","100"])
        run("export-sid-inputs",[sys.executable,"data/export_amazon23_embeddings.py","--catalog",str(a.catalog),
                                 "--shard_dir",str(shard_dir),"--output",str(a.matrix_output)])
        report["shard_dir"] = str(shard_dir.resolve())
        report["matrix_output"] = str(a.matrix_output.resolve())
        report["embedding_validation"] = json.loads(a.matrix_output.with_suffix(".manifest.json").read_text())
        report["status"] = "complete"; save()
        stats = json.loads((a.catalog.parent / "subset.stats.json").read_text())
        validation = report["embedding_validation"]
        unavailable = [dict(item_id=int(iid), parent_asin=item["parent_asin"],
                            reason=item.get("image_status"), error=item.get("image_error"))
                       for iid, item in catalog.items() if not item.get("images")]
        (a.output_dir / "image_unavailability.json").write_text(json.dumps(unavailable, indent=2))
        lines = [f"# {stats.get('dataset_name', 'Office_Products')} 商品属性与 embedding", "",
                 "状态：结构化属性和商品 embedding 已全部完成并通过核验。", "",
                 f"时间范围：{stats['start_time']} 至 {stats['cutoff_time']}（含边界）。",
                 f"商品 {stats['items']}，用户 {stats['users']}，交互 {stats['interactions']}。",
                 f"数据选取规则：{stats['selection']}。不因图片或文字资料缺失补入其他商品。", "",
                 f"实际有图商品 **{validation['joint_items']}**，无可用图商品 **{validation['text_items']}**。",
                 f"原始无图片 URL 商品 {stats['without_image_urls']}；其余图片不可用情况见 image_unavailability.json 和图片下载失败清单。",
                 "该子集明确包含两种输入方式：有图为图片＋结构化文字 joint，无图为同一个 embedding 模型的 text-only，逐商品记录 image_used 和 embedding_mode。", "",
                 f"无可用文字资料商品：{stats['without_usable_text']}；无法生成有效向量的异常：{validation['unembeddable_items']}。",
                 "异常仍保留在交互与 catalog 中，不以零向量伪造、不用后续商品替换。SID 输入附带原始商品 ID 映射；若有异常，不能声称全部商品已有 SID。", "",
                 f"训练/验证/测试交互数：{stats['split_counts']}；划分规则：{stats.get('split_policy', '全局时间80/10/10，边界并列时间进入较早划分')}。",
                 f"单次交互用户 {stats['singleton_users']}，均保留；首条交互对应空历史样本：{stats['empty_history_samples']}。",
                 "历史顺序以数据准备阶段的核验结果为准；样本仅包含该事件之前的历史，不引入未来交互。", "",
                 f"属性模型：{a.ollama_model}（Qwen3-VL-8B-Instruct Q4_K_M），Ollama，context={a.num_ctx}。",
                 f"Embedding 模型：{a.model.resolve()}（Qwen3-VL-Embedding-8B Q4_K_M）。",
                 f"投影：{a.mmproj.resolve()}；沿用已验证匹配的 projector（文件名 f16，实际 GGUF 元数据 BF16）。", "",
                 f"SID 输入矩阵：{a.matrix_output.resolve()}，形状 {validation['shape']}，数值全部有限。",
                 f"范数 min/mean/max：{validation['norm_min']:.8f}/{validation['norm_mean']:.8f}/{validation['norm_max']:.8f}。",
                 f"行到 item_id 映射：{validation['item_id_mapping']}。", "",
                 "耗时（秒）：", "```json", json.dumps(report['timings'],indent=2), "```", "",
                 f"本次启动时已完成属性记录 {report['attribute_resumed_successes']} 条，属性命令计时不包含这些已完成记录的历史耗时。", "",
                 "实际执行命令：", "```bash", *report['commands'].values(), "```", "",
                 f"数据与序列目录：{a.catalog.parent.resolve()}。",
                 f"属性、NPZ、运行日志与核验结果：{a.output_dir.resolve()}。",
                 "仅制作 SID 的商品表示和用户序列，未训练协同对齐、RFSQ、CPT、SFT 或 RL；向量成功不等于推荐效果已验证。"]
        (a.output_dir / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    except Exception as exc:
        report["status"] = "failed"; report["error"] = repr(exc); save(); raise
    finally:
        if server is not None: server.terminate(); server.wait(timeout=30)
        if gpu_lock is not None: gpu_lock.close()
    print("COMPLETE", str(report_path), flush=True)


if __name__ == "__main__": main()
