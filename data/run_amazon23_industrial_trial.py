"""Detached full official Industrial & Scientific 5-core representation trial."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args();os.chdir(ROOT)
    out=ROOT/'output/amazon23_industrial_5core'
    out.mkdir(parents=True,exist_ok=True)
    if a.detach:
        with (out/'background.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        (out/'pipeline.pid').write_text(str(proc.pid)+'\n')
        subprocess.Popen(['caffeinate','-i','-w',str(proc.pid)],stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        print('Background pipeline PID:',proc.pid);return
    dataset=ROOT/'data/Amazon23/Industrial_and_Scientific_5core'
    name='Industrial_and_Scientific'
    status=dict(status='running',stage='images',pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
                dataset=str(dataset),commands={},timings={})
    def save():
        temp=out/'trial.status.tmp';temp.write_text(json.dumps(status,ensure_ascii=False,indent=2));temp.replace(out/'trial.status.json')
    def run(phase,cmd):
        status['stage']=phase;status['commands'][phase]=shlex.join(cmd);save()
        start=time.perf_counter()
        with (out/f'{phase}.log').open('a') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        status['timings'][phase]=time.perf_counter()-start;save()
    save()
    try:
        local=dataset/f'{name}.local.item.jsonl'
        if not local.exists():
            run('images',[sys.executable,'data/download_amazon23_product_images.py','--catalog',str(dataset/f'{name}.item.jsonl'),
                '--output_catalog',str(local),'--image_dir',str(dataset/'images'),'--workers','16',
                '--allow_missing_images','--allow_unavailable_images'])
        matrix=dataset/f'{name}.emb-qwen-attributes-mixed-td.npy'
        if not matrix.exists():
            run('attributes-and-embeddings',[sys.executable,'data/run_amazon23_chronological_embeddings.py',
                '--catalog',str(local),'--output_dir',str(out/'representations'),
                '--ollama_model','benchmark-qwen3-vl:8b-instruct-q4_K_M',
                '--model','models/Qwen3-VL-Embedding-8B/Qwen3-VL-Embedding-8B-Q4_K_M.gguf',
                '--mmproj','models/Qwen3-VL-Embedding-8B/mmproj-Qwen3-VL-Embedding-8B-f16.gguf',
                '--llama_server','tools/llama.cpp/build/bin/llama-server','--matrix_output',str(matrix)])
        # Never continue RFSQ by substituting zero vectors for missing products.
        manifest=json.loads(matrix.with_suffix('.manifest.json').read_text())
        if manifest['unembeddable_items'] or manifest['embedded_items'] != manifest['catalog_items']:
            raise ValueError('Incomplete embeddings: exceptions retained in histories; SID training needs explicit handling')
        rfsq=out/'rfsq'
        if not (rfsq/'status.json').exists() or json.loads((rfsq/'status.json').read_text())['status']!='complete':
            if rfsq.exists():
                rfsq=out/f'rfsq-retry-{datetime.now():%Y%m%d-%H%M%S}'
            run('rfsq',[sys.executable,'data/run_amazon23_rfsq_benchmark.py','--dataset_dir',str(dataset),
                '--dataset_name',name,'--output_dir',str(rfsq),'--device','mps','--epochs','5000'])
        status['rfsq_output']=str(rfsq)
        previous=ROOT/'output/Office_Products_chrono5000-rfsq/benchmark-20261014-125441'
        new=json.loads((rfsq/'selected.index.metrics.json').read_text())['raw_comparison']
        old=json.loads((previous/'selected.index.metrics.json').read_text())['raw_comparison']
        stats=json.loads((dataset/'subset.stats.json').read_text())
        table=[]
        for label,key in [('商品数','items'),('原始SID超额重复比例','collision_rate')]:
            table.append(f'| {label} | {old[key]} | {new[key]} |')
        for i in range(3):
            table.append(f"| 第{i+1}层固定格子占用率 | {old['layers'][i]['occupancy_rate']:.4f} | {new['layers'][i]['occupancy_rate']:.4f} |")
        report='# Industrial & Scientific 官方去重5-core试验\n\n'
        report+=f"商品{stats['items']}、用户{stats['users']}、交互{stats['interactions']}，官方绝对时间划分{stats['split_counts']}。\n"
        report+=f"实际图文商品{manifest['joint_items']}、纯文本商品{manifest['text_items']}；完整向量与SID覆盖核验通过。\n\n"
        report+='| 指标 | Office chrono5000 | Industrial 官方5-core |\n|---|---:|---:|\n'+'\n'.join(table)
        report+='\n\nRFSQ没有可学习码本，统计的是固定离散格子及商品分组。使用同一训练配置，但类目、商品数、交互密度和时间范围都不同，不能把差异仅归因于5-core。详细指标、类别覆盖、重建、标量边界与消重对比见rfsq/REPORT.md。\n'
        report+='用户序列更丰富不等于推荐效果已经改善；本轮未训练推荐模型或协同对齐，也未运行CPT/SFT/RL。\n'
        (out/'REPORT.md').write_text(report)
        status.update(status='complete',stage='complete',finished_at=datetime.now().astimezone().isoformat());save()
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
