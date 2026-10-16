"""Run a lower/decaying-LR Office ablation without publishing experimental SIDs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args()
    out=ROOT/'output/Office_Products_chrono5000-rfsq/lr-linear-0003'
    out.mkdir(parents=True,exist_ok=False)
    if a.detach:
        with (out/'background.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run_dir',str(out)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
        (out/'pipeline.pid').write_text(str(process.pid)+'\n')
        subprocess.Popen(['caffeinate','-i','-w',str(process.pid)],stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        print('Background LR trial PID:',process.pid)
        return
    run(out)


def run(out):
    baseline=ROOT/'output/Office_Products_chrono5000-rfsq/benchmark-20261014-125441'
    status=dict(status='running',started_at=datetime.now().astimezone().isoformat(),baseline=str(baseline),
                lr=0.0003,scheduler='linear',publish_index=False)
    def save(): (out/'trial.status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    save()
    try:
        with (out/'training-pipeline.log').open('w') as log:
            subprocess.run([sys.executable,'data/run_amazon23_rfsq_benchmark.py','--output_dir',str(out/'benchmark'),
                '--device','mps','--epochs','5000','--lr','0.0003','--lr_scheduler_type','linear',
                '--no_publish_index'],check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        old=json.loads((baseline/'status.json').read_text())
        new=json.loads((out/'benchmark/status.json').read_text())
        if old['matrix_sha256']!=new['matrix_sha256']:raise ValueError('Input embeddings differ from baseline')
        reports=[json.loads((path/'selected.index.metrics.json').read_text())['raw_comparison']
                 for path in [baseline,out/'benchmark']]
        metrics=[dict(raw_collision_rate=r['collision_rate'],exclusive_item_rate=r['full_sid']['exclusive_item_rate'],
                 layer_occupancy=[v['occupancy_rate'] for v in r['layers']],
                 non_singleton_purity=r['full_sid']['purity']['non_singleton_cluster_item_weighted_purity'],
                 **s['reconstruction']['selected']) for r,s in zip(reports,[old,new])]
        status.update(status='complete',baseline_metrics=metrics[0],experiment_metrics=metrics[1],
                      input_hash_match=True,finished_at=datetime.now().astimezone().isoformat())
        save()
        rows=[]
        for label,key in [('原始SID超额重复比例','raw_collision_rate'),('独占商品比例','exclusive_item_rate'),
                          ('标准化空间重建MSE','standardized_reconstruction_mse'),
                          ('原空间平均重建余弦','original_space_cosine_mean'),
                          ('非单商品簇类别纯度','non_singleton_purity')]:
            fmt=lambda v:'N/A' if v is None else f'{v:.6f}'
            rows.append(f'| {label} | {fmt(metrics[0][key])} | {fmt(metrics[1][key])} |')
        text='# Office chrono5000 学习率策略对照\n\n'
        text+='实验完成。原始embedding逐字节hash一致，用户交互/类别/网络/格子等级/残差scale配置/seed/训练轮数保持一致。\n'
        text+='基线：lr=0.001，constant；实验：lr=0.0003，linear，预热50轮后逐步衰减至0。两者选模规则相同。\n'
        text+='同时改变初始学习率和衰减策略，因此这里只评价整个策略，不能区分两者各自的贡献。\n\n'
        text+='| 指标（比例为0–1） | 原基线 | 新学习率策略 |\n|---|---:|---:|\n'+'\n'.join(rows)
        text+='\n\n不将消重后唯一率作为质量提升依据；原始重复、重建和语义分组可能有取舍。完整指标、格子饱和、best_loss模型和序列核验见benchmark/REPORT.md。\n'
        text+='实验SID仅保存在本输出目录，未替换数据目录的Office_Products.index.json。未改变Industrial数据准备任务，未训练推荐模型。\n'
        (out/'REPORT.md').write_text(text)
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':
    if '--run_dir' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--run_dir',type=Path,required=True);run(p.parse_args().run_dir)
    else:main()
