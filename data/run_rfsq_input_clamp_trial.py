"""Office5000: historical input-clamp behavior followed by collision fine-tuning."""
import argparse
from datetime import datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def run(out, dataset, name, matrix, base_epochs=5000, micro_epochs=2000, resume=None, patience=0):
    status=dict(dataset_dir=str(dataset),dataset_name=name,matrix_file=str(matrix),status='running',stage='base-training',commands={},timings={})
    def save(): (out/'trial.status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    def command(stage,args):
        status['stage']=stage;status['commands'][stage]=args;save();start=time.perf_counter()
        with (out/f'{stage}.log').open('w') as log:subprocess.run(args,check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status['timings'][stage]=time.perf_counter()-start;save()
    save()
    try:
        base_command=[sys.executable,'data/run_amazon23_rfsq_benchmark.py','--output_dir',str(out/'base'),
            '--dataset_dir',str(dataset),'--dataset_name',name,'--embedding_file',str(matrix),
            '--latent_activation','none','--quantizer_input_clamp','--no_publish_index','--device','mps','--epochs',str(base_epochs)]
        if resume: base_command.extend(['--resume_from_checkpoint',resume])
        command('base-training',base_command)
        base=json.loads((out/'base/status.json').read_text())
        checkpoint=Path(base['checkpoint_dir'])/'best_collision_model.pth'
        before=json.loads((out/'base/selected.index.metrics.json').read_text())['raw_comparison']
        if before['collision_rate'] >= .8:
            status.update(status='complete',stage='complete',micro_skipped='Base model has >=80% excess collisions; avoid quadratic pair explosion')
            (out/'REPORT.md').write_text(f"# 输入边界对照\n\n基础训练已完成，但原始重复率为{before['collision_rate']:.2%}，未启动碰撞微调以避免大规模无效配对。原SID未替换。详见base/REPORT.md。\n")
            save();return
        with (ROOT/'output/.local-model-gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            command('collision-finetune',[sys.executable,'rq/finetune_rfsq_collisions.py','--checkpoint',str(checkpoint),
                '--output_dir',str(out/'finetune'),'--device','mps','--epochs',str(micro_epochs),'--early_stop_patience',str(patience),'--learning_rate','0.0001',
                '--collision_weight','0.3','--margin','1.0','--batch_size','2048','--pair_batch_size','128'])
            selected=out/'finetune/best_collision_model.pth'
            command('export-finetuned',[sys.executable,'rq/generate_rfsq_indices.py','--checkpoint',str(selected),
                '--output_file',str(out/'finetune/raw.index.json'),'--device','mps',
                '--item_file',str(dataset/f'{name}.item.json'),
                '--interaction_file',str(dataset/f'{name}.interactions.jsonl')])
            spec=importlib.util.spec_from_file_location('benchmark',ROOT/'data/run_amazon23_rfsq_benchmark.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            rec=module.reconstruction(selected,'mps')
        after=json.loads((out/'finetune/raw.index.metrics.json').read_text())
        expected_items=json.loads((dataset/'subset.stats.json').read_text())['items']
        if after['items']!=expected_items or rec['items']!=expected_items:raise ValueError('Incomplete item coverage')
        command('export-unique',[sys.executable,'rq/generate_rfsq_indices.py','--checkpoint',str(selected),
            '--output_file',str(out/'finetune/unique.index.json'),'--device','mps','--repair_collisions',
            '--item_file',str(dataset/f'{name}.item.json'),
            '--interaction_file',str(dataset/f'{name}.interactions.jsonl')])
        unique=json.loads((out/'finetune/unique.index.json').read_text())
        assert set(unique)==set(map(str,range(expected_items)))
        assert len({tuple(v) for v in unique.values()})==expected_items
        # Preserve the ASIN-to-row identity for downstream consumers.
        item_meta=json.loads((dataset/f'{name}.item.json').read_text())
        asin_index={item_meta[i]['parent_asin']:sid for i,sid in unique.items()}
        (out/'finetune/unique.asin.index.json').write_text(json.dumps(asin_index))
        stages=[('量化前clamp基础训练',before,base['reconstruction']['selected']),('碰撞微调后',after,rec)]
        if name=='Office_Products':
            oldpath=ROOT/'output/Office_Products_chrono5000-rfsq/benchmark-20261014-125441'
            old=json.loads((oldpath/'selected.index.metrics.json').read_text())['raw_comparison']
            oldrec=json.loads((oldpath/'status.json').read_text())['reconstruction']['selected']
            stages.insert(0,('原基线',old,oldrec))
        lines=[f'# {name} 输入边界与碰撞微调实验','',
            '输入、商品集合与用户序列保持一致，原SID不替换；原始指标不包含编号消重。',
            '基础训练：none激活，每层FSQ输入clamp(-1,1)，取整前范围限制参与自动求导；其余配置沿用原基线。',
            f'微调：L=MSE+0.3*mean(relu(1-L2距离))；lr=1e-4、AdamW默认weight_decay=0.01、最多{micro_epochs}轮，连续{patience}次评估无碰撞或重建改善时停止；每20轮重挖、25轮评估；相同输入不配对。',
            '起始checkpoint也参与选模，防止微调退化时自动选用更差模型。','',
            '| 阶段 | 原始重复率 | 第二层占用率 | 重建MSE | 平均重建余弦 | 非单商品簇纯度 |',
            '|---|---:|---:|---:|---:|---:|']
        for label,m,r in stages:
            purity=m['full_sid']['purity']['non_singleton_cluster_item_weighted_purity']
            lines.append(f"| {label} | {m['collision_rate']:.2%} | {m['layers'][1]['occupancy_rate']:.2%} | {r['standardized_reconstruction_mse']:.6f} | {r['original_space_cosine_mean']:.6f} | {purity if purity is not None else 'N/A'} |")
        lines+=['','基础训练同时改变激活与量化前边界，因此此阶段是历史行为组合对照，不分别归因；微调前后使用相同边界规则。',
            '原始SID见finetune/raw.index.json；唯一化版本见unique.index.json及unique.asin.index.json。消重是明确后处理，不作为原始量化质量指标。',
            '完整指标见base/REPORT.md及finetune/raw.index.metrics.json。尚未评估推荐效果。']
        (out/'REPORT.md').write_text('\n'.join(lines))
        status.update(status='complete',stage='complete',before_collision=before['collision_rate'],
            after_collision=after['collision_rate'],reconstruction=rec,validated_items=expected_items,
            unique_index_file=str(out/'finetune/unique.index.json'));save()
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run_dir',type=Path)
    p.add_argument('--dataset_dir',type=Path,default=ROOT/'data/Amazon23/Office_Products_chrono5000')
    p.add_argument('--dataset_name',default='Office_Products')
    p.add_argument('--embedding_file',type=Path)
    p.add_argument('--base_epochs',type=int,default=5000)
    p.add_argument('--micro_epochs',type=int,default=2000)
    p.add_argument('--resume_from_checkpoint')
    p.add_argument('--early_stop_patience',type=int,default=0)
    a=p.parse_args();dataset=a.dataset_dir.resolve();name=a.dataset_name
    matrix=a.embedding_file.resolve() if a.embedding_file else dataset/f'{name}.emb-qwen-attributes-mixed-td.npy'
    if a.run_dir:run(a.run_dir,dataset,name,matrix,a.base_epochs,a.micro_epochs,a.resume_from_checkpoint,a.early_stop_patience)
    else:
        out=ROOT/f'output/{dataset.name}-rfsq/input-clamp-collision-{datetime.now():%Y%m%d-%H%M%S}'
        out.mkdir(parents=True)
        with (out/'background.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run_dir',str(out),'--dataset_dir',str(dataset),
                '--dataset_name',name,'--embedding_file',str(matrix),'--base_epochs',str(a.base_epochs),
                '--micro_epochs',str(a.micro_epochs),'--early_stop_patience',str(a.early_stop_patience),
                *(['--resume_from_checkpoint',a.resume_from_checkpoint] if a.resume_from_checkpoint else [])],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'pipeline.pid').write_text(str(proc.pid))
        subprocess.Popen(['caffeinate','-i','-w',str(proc.pid)],stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        print('Background trial:',out,'PID:',proc.pid)
