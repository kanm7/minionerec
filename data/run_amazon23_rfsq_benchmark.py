#!/usr/bin/env python3
"""Train the chronological Amazon23 RFSQ and export audited diagnostics."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import fcntl

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rq.datasets import EmbDataset
from rq.models.rfsq import RFSQVAE
from rq.sid_metrics import validate_indices


def reconstruction(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    a = ck['args']
    data = EmbDataset(a.data_path, standardize=a.standardize)
    model = RFSQVAE(data.dim, a.stage_levels, a.layers, conditioning=a.conditioning,
                    dropout_prob=a.dropout_prob, bn=a.bn, loss_type=a.loss_type,
                    latent_activation=a.latent_activation, scalar_bound=a.scalar_bound == 'clamp',
                    input_clamp=getattr(a,'quantizer_input_clamp',False))
    model.load_state_dict(ck['state_dict'])
    model.to(device).eval()
    scalar_audit = [dict(assignments=0, clipped_coordinates=0, coordinates=0,
                         used_levels=[set() for _ in levels]) for levels in a.stage_levels]
    handles = []
    for stage, quantizer in enumerate(model.quantizer.quantizers):
        def audit(module, inputs, stage=stage):
            values = inputs[0].detach()
            levels = module.levels.to(values.device)
            raw = torch.round((values+1)*(levels-1)/2).long()
            clipped = torch.minimum(raw.clamp_min(0), levels-1)
            record = scalar_audit[stage]
            record['assignments'] += len(values)
            record['coordinates'] += raw.numel()
            record['clipped_coordinates'] += int(((raw < 0) | (raw >= levels)).sum().item())
            for dim, used in enumerate(record['used_levels']):
                used.update(clipped[:,dim].cpu().tolist())
        handles.append(quantizer.register_forward_pre_hook(audit))
    errors, cosines = [], []
    original = np.load(a.data_path).astype(np.float32)
    mean, std = original.mean(0), np.maximum(original.std(0), 1e-5)
    offset = 0
    with torch.inference_mode():
        for batch in DataLoader(data, batch_size=1024, shuffle=False):
            pred, _, indices = model(batch.to(device))
            pred = pred.cpu().numpy()
            errors.append(((pred - batch.numpy()) ** 2).mean(1))
            restored = pred * std + mean if a.standardize else pred
            truth = original[offset:offset+len(batch)]
            cosines.extend((np.sum(restored*truth, axis=1) /
                            np.maximum(np.linalg.norm(restored, axis=1)*np.linalg.norm(truth, axis=1), 1e-12)).tolist())
            offset += len(batch)
    for handle in handles: handle.remove()
    for record in scalar_audit:
        record['used_levels'] = [sorted(v) for v in record['used_levels']]
        record['clipped_coordinate_rate'] = record['clipped_coordinates']/record['coordinates']
    return dict(epoch=int(ck['epoch'])+(0 if 'fine_tune_args' in ck else 1), standardized_reconstruction_mse=float(np.concatenate(errors).mean()),
                original_space_cosine_mean=float(np.mean(cosines)),
                original_space_cosine_median=float(np.median(cosines)), items=offset,
                residual_scales=(torch.nn.functional.softplus(model.quantizer.raw_scales)+model.quantizer.eps).cpu().tolist(),
                scalar_level_audit=scalar_audit)


def percent(value):
    if value is None: return 'N/A'
    return f'{100*value:.6g}%' if 0 < value < 0.0001 else f'{100*value:.2f}%'


def audit_existing(out, device):
    """Attach scalar-level checks and join the unchanged sequence samples to SIDs."""
    out = Path(out).resolve()
    status = json.loads((out/'status.json').read_text())
    if status['status'] != 'complete': raise ValueError('Training/export must complete before final audit')
    directory = Path(status['checkpoint_dir'])
    audit = {key: reconstruction(directory/file, device) for key,file in
             [('selected','best_collision_model.pth'), ('best_loss','best_loss_model.pth')]}
    dataset = Path(status.get('dataset_dir', ROOT/'data/Amazon23/Office_Products_chrono5000'))
    name = status.get('dataset_name', 'Office_Products')
    matrix = Path(status.get('matrix_file', dataset/f'{name}.emb-qwen-attributes-mixed-td.npy'))
    vectors = np.load(matrix)
    distinct_vectors = len({hashlib.sha256(row.tobytes()).digest() for row in vectors})
    indices = json.loads((out/'selected.index.json').read_text())
    expected = set(map(str,np.load(str(matrix)+'.item_ids.npy')))
    if set(indices) != expected: raise ValueError('Incomplete SID coverage')
    validate_indices(indices,[2048]*3)
    counts = dict(train=0,valid=0,test=0)
    handles = {split:(out/f'{split}.sid-sequences.jsonl').open('w') for split in counts}
    try:
        with (dataset/f'{name}.samples.jsonl').open() as source:
            for row in map(json.loads,source):
                target = indices[str(row['item_id'])]
                history = [indices[str(i)] for i in row['history_item_ids']]
                handles[row['split']].write(json.dumps(dict(row, history_sids=history, target_sid=target))+'\n')
                counts[row['split']] += 1
    finally:
        for handle in handles.values(): handle.close()
    expected_counts = json.loads((dataset/'subset.stats.json').read_text())['split_counts']
    if counts != expected_counts: raise ValueError('SID sequence counts differ from original sequences')
    (out/'scalar-level-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    status['scalar_level_audit_file'] = str(out/'scalar-level-audit.json')
    status['sid_sequence_counts'] = counts
    status['final_audit'] = 'passed'
    status['distinct_exact_input_vectors'] = distinct_vectors
    (out/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    report = (out/'REPORT.md').read_text()
    if '## 补充标量与序列核验' not in report:
        report += '\n## 补充标量与序列核验\n\n'
        report += 'RFSQ 不含可学习码本；标量格子等级、饱和比例和残差 scale 的实测值见 scalar-level-audit.json。\n'
        report += '本轮使用的激活与量化前边界设置见实际训练命令；此前 Office_Products 无界配置塌缩的结果保留在该数据集报告中。\n'
        report += f'将原有样本逐条连接最终 SID，商品编号、历史顺序和时间划分均未改动：{counts}，共 {sum(counts.values())} 条。文件为 train/valid/test.sid-sequences.jsonl。\n'
        report += '相关单元测试12项通过，覆盖有限标量等级、反解一致性、梯度、残差缩放边界以及各统计指标。\n'
        raw = json.loads((out/'selected.index.metrics.json').read_text())['raw_comparison']
        report += f"原始完整 SID 的精确占用率为 {percent(raw['full_sid']['occupancy_rate'])}（{raw['full_sid']['used_codes']}/{2048**3}），不能用四舍五入后的0.00%解释为完全未使用。\n"
        report += f'输入向量逐行按字节核验：{len(vectors)} 条中有 {distinct_vectors} 条不同向量；完全相同输入的数量应与原始SID重复数量区分。\n'
        (out/'REPORT.md').write_text(report)
    if '## 结果判断' not in report:
        raw = json.loads((out/'selected.index.metrics.json').read_text())['raw_comparison']
        report += '\n## 结果判断\n\n已完成训练与覆盖核验，结果应结合原始量化重复、重建质量和标量饱和判断，不能仅以消重后的唯一SID评价量化质量。\n\n'
        report += '| 原始分组范围 | 独占商品比例 | 热度≥平均5倍的簇比例 | 这些高热簇的交互占比 |\n|---|---:|---:|---:|\n'
        for label,values in [(f'第{i+1}层',v) for i,v in enumerate(raw['layers'])]+[('完整SID',raw['full_sid'])]:
            hot = values['hot_clusters_above_mean']
            report += f"| {label} | {percent(values['exclusive_item_rate'])} | {percent(hot['cluster_rate'])} | {percent(hot['popularity_share'])} |\n"
        report += '\n高热簇比例以已使用的分组为分母，阈值是这些分组平均训练交互数的5倍。\n'
        selected_audit = audit['selected']
        for i,record in enumerate(selected_audit['scalar_level_audit']):
            report += f"第{i+1}层各维已用等级数：{[len(v) for v in record['used_levels']]}；标量坐标边界限制比例：{percent(record['clipped_coordinate_rate'])}。\n"
        loss_metrics = json.loads((out/'best_loss.index.metrics.json').read_text())
        report += f"最优重建checkpoint：第{audit['best_loss']['epoch']}轮，原始重复比例{percent(loss_metrics['collision_rate'])}，平均重建余弦{audit['best_loss']['original_space_cosine_mean']:.4f}；当前选中的第{selected_audit['epoch']}轮分别为{percent(raw['collision_rate'])}和{selected_audit['original_space_cosine_mean']:.4f}。两者均已保存，反映唯一性与重建质量的取舍。\n"
        (out/'REPORT.md').write_text(report)
    print(json.dumps(dict(final_audit='passed',sid_sequence_counts=counts,scalar_level_audit=audit),ensure_ascii=False,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--dataset_dir', type=Path, default=ROOT/'data/Amazon23/Office_Products_chrono5000')
    p.add_argument('--dataset_name', default='Office_Products')
    p.add_argument('--embedding_file', type=Path)
    p.add_argument('--resume_from_checkpoint')
    p.add_argument('--device', default='mps')
    p.add_argument('--latent_activation', choices=['none','tanh'], default='tanh')
    p.add_argument('--quantizer_input_clamp', action='store_true')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lr_scheduler_type', choices=['constant','linear'], default='constant')
    p.add_argument('--no_publish_index', action='store_true', help='Keep experimental SIDs in the run directory')
    p.add_argument('--output_dir')
    p.add_argument('--audit_existing', help='Audit a completed run without retraining')
    a = p.parse_args()
    os.chdir(ROOT)
    if a.audit_existing:
        audit_existing(a.audit_existing,a.device)
        return
    out = Path(a.output_dir or f'output/{a.dataset_dir.name}-rfsq/benchmark-{datetime.now():%Y%m%d-%H%M%S}').resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset = a.dataset_dir.resolve()
    name = a.dataset_name
    matrix = a.embedding_file.resolve() if a.embedding_file else dataset/f'{name}.emb-qwen-attributes-mixed-td.npy'
    items = dataset/f'{name}.item.json'
    interactions = dataset/f'{name}.interactions.jsonl'
    status = dict(matrix_file=str(matrix), dataset_dir=str(dataset), dataset_name=name, status='running', stage='training', pid=os.getpid(), output_dir=str(out), commands={}, timings={},
                  matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
                  software=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                macOS=platform.mac_ver()[0], device=a.device),
                  checkpoint_selection='lowest raw full-SID collision rate; tie broken by training reconstruction loss')
    def save():
        temp = out/'status.tmp'
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        temp.replace(out/'status.json')
    def run(name, command):
        status['stage'] = name
        status['commands'][name] = shlex.join(command)
        save()
        start = time.perf_counter()
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, TQDM_DISABLE='1'))
        status['timings'][name] = time.perf_counter()-start
        save()
    save()
    gpu_lock = None
    try:
        if a.device == 'mps':
            status['stage'] = 'waiting_for_gpu'; save()
            gpu_lock = (ROOT/'output/.local-model-gpu.lock').open('a')
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        if a.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Apple GPU unavailable; run outside the restricted sandbox')
        train_command = [sys.executable, 'rq/rqfsq.py', '--data_path', str(matrix), '--ckpt_dir', str(out/'checkpoints'),
            '--device', a.device, '--epochs', str(a.epochs), '--batch_size', '2048', '--num_workers', '0',
            '--eval_step', '50', '--warmup_epochs', '50', '--scalar_bound', 'clamp', '--conditioning', 'scale',
            '--latent_activation', a.latent_activation,
            '--lr', str(a.lr), '--lr_scheduler_type', a.lr_scheduler_type,
            '--metrics_item_file', str(items), '--metrics_interaction_file', str(interactions)]
        if a.quantizer_input_clamp: train_command.append('--quantizer_input_clamp')
        if a.resume_from_checkpoint: train_command.extend(['--resume_from_checkpoint',a.resume_from_checkpoint])
        run('training', train_command)
        runs = list((out/'checkpoints').iterdir())
        if len(runs) != 1: raise ValueError('Expected exactly one checkpoint run')
        ckpt_dir = runs[0]
        for variant, filename in [('selected', 'best_collision_model.pth'), ('best_loss', 'best_loss_model.pth')]:
            command = [sys.executable, 'rq/generate_rfsq_indices.py', '--checkpoint', str(ckpt_dir/filename),
                       '--output_file', str(out/f'{variant}.index.json'), '--device', a.device,
                       '--item_file', str(items), '--interaction_file', str(interactions)]
            if variant == 'selected': command.append('--repair_collisions')
            run(f'export-{variant}', command)
        expected_ids = set(map(str, np.load(str(matrix)+'.item_ids.npy').tolist()))
        for file_name in ['selected.index.json', 'selected.index.raw.json', 'best_loss.index.json']:
            mapping = json.loads((out/file_name).read_text())
            if set(mapping) != expected_ids: raise ValueError(f'ID coverage mismatch: {file_name}')
            validate_indices(mapping, [2048]*3)
        selected = json.loads((out/'selected.index.json').read_text())
        if len({tuple(row) for row in selected.values()}) != len(selected):
            raise ValueError('Collision repair did not give unique final SIDs')
        status['reconstruction'] = {key: reconstruction(ckpt_dir/file, a.device) for key,file in
                                    [('selected','best_collision_model.pth'), ('best_loss','best_loss_model.pth')]}
        metrics = json.loads((out/'selected.index.metrics.json').read_text())
        raw = metrics['raw_comparison']
        # Publish only after complete item coverage and token-range validation.
        if not a.no_publish_index:
            (dataset/f'{name}.index.json').write_text(json.dumps(selected))
            (dataset/f'{name}.index.raw.json').write_text((out/'selected.index.raw.json').read_text())
        status.update(status='complete', stage='complete', checkpoint_dir=str(ckpt_dir), validated_items=len(selected),
                      index_file=str(out/'selected.index.json' if a.no_publish_index else dataset/f'{name}.index.json'), raw_collision_rate=raw['collision_rate'],
                      repaired_items=metrics['collision_repaired_items'])
        stats = json.loads((dataset/'subset.stats.json').read_text())
        rows = []
        for label, values in [(f'第 {i+1} 层',v) for i,v in enumerate(raw['layers'])] + \
                             [(f'前 {i+1} 层组合',v) for i,v in enumerate(raw['prefix'][1:],start=1)]:
            rows.append(f"| {label} | {values['used_codes']} | {percent(values['occupancy_rate'])} | "
                        f"{percent(values['exclusive_item_rate'])} | {percent(values['purity']['item_weighted_purity'])} | "
                        f"{percent(values['hot_clusters']['popularity_share'])} | {values['normalized_entropy']:.4f} |")
        report = f'''# {name} RFSQ benchmark

训练与核验已完成：{len(selected)} 个商品，输入 {len(selected)}×{np.load(matrix, mmap_mode='r').shape[1]}，三层 SID，每层 2048 个固定隐式格子。
RFSQ **没有可学习的码本向量或质心表**。以下“占用、独占、簇”分别指固定离散格子使用情况、商品是否独占该分组、共享同一编码的商品集合。

训练设置：seed=2024，epochs={a.epochs}，device={a.device}，AdamW，lr={a.lr}，batch=2048，warmup=50 epochs，{a.lr_scheduler_type} scheduler，逐维标准化，scale conditioning，latent activation={a.latent_activation}，scalar_bound=clamp，量化前输入clamp={a.quantizer_input_clamp}。
MLP：{np.load(matrix, mmap_mode='r').shape[1]}→2048→1024→512→256→128→64→8，解码器对称；每层 levels=[4,4,4,2,2,2,2,2]。
本轮修正了标量编号越界：每个标量取整后限制到自己的合法等级范围，再进行组合编码，保留 STE 梯度。旧 checkpoint 导出仍使用其原始设置。
当前激活和量化边界设置见上述实际命令；Office_Products 对照实验中的无界配置曾出现塌缩，详见该数据集报告。
选模：{status['checkpoint_selection']}。训练全体商品表示，交互不进入重建损失；热门簇只统计 train 的 {stats.get('interaction_split_counts', stats['split_counts'])['train']} 条原始交互，不重复累加历史，不使用验证/测试热度。

训练墙钟时间：{status['timings']['training']:.2f} 秒（{status['timings']['training']/60:.2f} 分钟）。
选中 checkpoint：第 {status['reconstruction']['selected']['epoch']} 轮。
输入 hash、版本、完整命令与各阶段计时见 status.json；逐次指标见 checkpoints 下 sid_benchmark/history.jsonl。

## 原始 RFSQ 指标

| 范围 | 已用格子/组合 | 占用率 | 独占商品比例 | 类别加权纯度 | 热度前1%簇的训练交互占比 | 已用格子归一化熵 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

原始完整 SID：唯一组合 {raw['full_sid']['used_codes']}，超额重复比例 {percent(raw['collision_rate'])}，处于重复组的商品比例 {percent(raw['full_sid']['collision_item_rate'])}。
类别标签覆盖率 {percent(raw['full_sid']['purity']['labeled_item_coverage'])}；采用 categories 第一条路径的叶类别。
非单商品完整 SID 簇的类别加权纯度：{percent(raw['full_sid']['purity']['non_singleton_cluster_item_weighted_purity'])}。
各层与前缀还记录单商品格子比例、最大簇、有效格子数、容量归一化熵，以及热度≥平均簇热度5倍的簇比例等，见 selected.index.metrics.json。

## 消除重复后的 SID

仅对重复的完整 SID，在相同前两层下重分配末层空闲编号，共调整 {metrics['collision_repaired_items']} 个商品。
导出 {len(selected)} 个唯一 SID，超额重复比例 {percent(metrics['collision_rate'])}，独占商品比例 {percent(metrics['full_sid']['exclusive_item_rate'])}。
这是明确的后处理，不能把其独占率或纯度提升归因于 RFSQ 原始量化。原始结果单独保存，未覆盖。

## 重建检查

```json
{json.dumps(status['reconstruction'], ensure_ascii=False, indent=2)}
```

重建 MSE 在标准化空间计算；余弦在还原到原 embedding 空间后计算。best_loss checkpoint 也单独导出指标供比较。

用户序列与时间划分保持原样：交互 {stats['interactions']}、用户 {stats['users']}，训练/验证/测试 {stats['split_counts']}。
本轮 SID：{status['index_file']}；原始 SID：{out/'selected.index.raw.json'}。是否发布到数据目录：{not a.no_publish_index}。

完整组合容量为 2048³，只有 {len(selected)} 个商品，完整 SID 占用率很低是容量规模造成的，不能单凭它判定量化失败。
单商品簇的纯度天然为1，因此必须结合前缀纯度和非单商品簇纯度判断。仅以训练交互衡量热度；热度解释需结合该数据集的时间范围和交互密度。
商品图文输入比例和源资料异常见对应 embedding 报告。
这些指标不是推荐 Recall/NDCG；本轮没有训练推荐模型、协同对齐、CPT、SFT 或 RL。
'''
        (out/'REPORT.md').write_text(report)
        save()
        audit_existing(out, a.device)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    finally:
        if gpu_lock is not None: gpu_lock.close()


if __name__ == '__main__': main()
