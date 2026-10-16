"""Plot the completed Industrial CPT checkpoint lineage; exclude rolled-back updates."""
from pathlib import Path
import csv
import json
import os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/Industrial_CPT_loss_curves'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

RUNS=[('Industrial_CPT_pure_sid',0,250),('Industrial_CPT_pure_sid_resume',250,1250),
 ('Industrial_CPT_pure_sid_resume2',1250,2500),('Industrial_CPT_pure_sid_batch4',2500,2750),
 ('Industrial_CPT_pure_sid_resume2750',2750,4063)]
train={};validation={};full=None
for name,start,end in RUNS:
 current=start
 for line in (ROOT/'output'/name/'metrics.jsonl').read_text().splitlines():
  record=json.loads(line)
  if 'step' in record:
   current=record['step']
   if start<current<=end:train[current]=dict(step=current,loss=record['train_loss'],source_run=name)
  elif record.get('scope') in ('initial_validation_subset','validation_subset','final_validation_subset'):
   if current<=end:
    # Keep original boundary evaluation instead of rounding-only repeated resume checks.
    validation.setdefault(current,dict(step=current,full_sequence_loss=record['full_sequence_loss'],
      next_item_sid_loss=record['next_item_sid_loss'],samples=record['samples'],source_run=name))
  elif record.get('scope')=='selected_full_validation':full=record
assert sorted(train)==list(range(1,4064)), 'Missing or duplicate lineage coverage'
assert sorted(validation)==list(range(0,4001,250))+[4063]
assert {r['samples'] for r in validation.values()}=={512}
best=min(validation.values(),key=lambda r:r['next_item_sid_loss'])
assert best['step']==3750
for filename,rows in [('training_loss.csv',list(train.values())),('validation_loss.csv',list(validation.values()))]:
 with (OUT/filename).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'selected_full_validation.json').write_text(json.dumps(dict(checkpoint_step=best['step'],**full),indent=2))
fontpath=Path('/System/Library/Fonts/STHeiti Light.ttc')
if fontpath.exists():
 prop=FontProperties(fname=str(fontpath));plt.rcParams['font.family']=prop.get_name()
 from matplotlib import font_manager
 font_manager.fontManager.addfont(str(fontpath))
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
x=np.array(sorted(train));y=np.array([train[int(i)]['loss'] for i in x])
rolling=np.array([y[max(0,i-49):i+1].mean() for i in range(len(y))])
vx=np.array(sorted(validation));seq=np.array([validation[int(i)]['full_sequence_loss'] for i in vx]);target=np.array([validation[int(i)]['next_item_sid_loss'] for i in vx])
fig,axes=plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Industrial & Scientific · CPT loss 曲线',fontsize=18,fontweight='bold',y=.98)
ax=axes[0]
ax.plot(x,y,color='#2766b5',alpha=.20,lw=.65,label='训练：每步整序列 loss')
ax.plot(x,rolling,color='#164d93',lw=1.9,label='训练：过去 50 步移动平均')
ax.set_title('训练集',fontsize=13,pad=12)
ax.set_ylim(bottom=0);ax.legend(frameon=False,loc='upper right')
ax=axes[1]
ax.plot(vx,seq,'o-',color='#16836e',lw=1.8,ms=4,label='验证：整序列 loss（512 条）')
ax.plot(vx,target,'o-',color='#d67c16',lw=1.8,ms=4,label='验证：目标商品 3 个 SID loss（512 条）')
ax.scatter([best['step']],[full['next_item_sid_loss']],marker='*',s=160,color='#8b3ab0',zorder=5,label='最佳模型全量验证：4.073（50,985 条）')
ax.annotate(f"最佳子集 loss {best['next_item_sid_loss']:.3f}\nstep {best['step']}",xy=(best['step'],best['next_item_sid_loss']),xytext=(2280,6.0),fontsize=9,
 arrowprops=dict(arrowstyle='->',color='#a46013'),color='#995500')
ax.set_title('验证集：两种计算口径',fontsize=13,pad=12)
ax.legend(frameon=False,loc='upper right',fontsize=8.8)
ax.set_ylim(0,14.6)
for ax in axes:
 ax.set_xlabel('优化器更新步数');ax.set_ylabel('交叉熵 loss');ax.grid(alpha=.16)
 ax.set_xlim(0,4100);ax.set_xticks([0,1000,2000,3000,4000])
 ax.axvline(2500,color='#777',ls=':',lw=.9,alpha=.65)
fig.text(.5,.075,'训练曲线已按检查点续训路径拼接，回滚后被丢弃的步骤不计入；点线为固定 512 条验证样本。',ha='center',fontsize=9,color='#555')
fig.text(.5,.038,'训练 loss 覆盖整条序列；目标商品 loss 只覆盖最后 3 个 SID token，二者不能直接比较。星号为独立全量验证。',ha='center',fontsize=9,color='#555')
fig.subplots_adjust(left=.065,right=.985,top=.87,bottom=.18,wspace=.22)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'cpt_loss_curves.{ext}',dpi=220,facecolor='white')
plt.close(fig)
(OUT/'README.md').write_text(f'''# CPT loss curves

All 4,063 optimizer updates follow the final checkpoint lineage. Cutoffs: original 1–250; resume 251–1250; resume2 1251–2500; batch4 2501–2750; resume2750 2751–4063. Rolled-back updates, repeated boundary validation and the preflight loss are excluded.

Training: per-update token-weighted full-sequence loss, raw and trailing 50-update arithmetic mean. A batch minimum is not the whole-training-set loss.

Validation: 18 measurements, the same first 512 examples, with separate full-sequence and target-three-SID losses. The best target-SID validation loss is {best['next_item_sid_loss']:.6f} at step {best['step']}. Last-step validation: full sequence {seq[-1]:.6f}, target SID {target[-1]:.6f}. No interpolated points are measured results.

The purple star evaluates that selected checkpoint on all {full['samples']:,} validation examples: target SID {full['next_item_sid_loss']:.6f}, full sequence {full['full_sequence_loss']:.6f}. It is not another point in the fixed-subset curve. Checkpoint selection uses the 512-example subset. Test data is unused. The dotted vertical line at 2500 marks the microbatch switch (2 to 4; effective batch remains 64); earlier original segment used microbatch 4.

Artifacts: PNG for viewing, PDF/SVG for export, training/validation CSV for reproducibility, selected_full_validation.json for the independent full-validation result. These are loss curves, not recommendation Recall/NDCG.
''')
print(json.dumps(dict(directory=str(OUT),training_points=len(train),validation_points=len(validation),best_step=best['step']),ensure_ascii=False))
