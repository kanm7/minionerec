"""Dependency-light equivalents of rl.py's ranking rewards and ReReTrainer GRPO loss."""
import math
import torch

def ranking_rewards(completions,targets,num_generations=16):
 if len(completions)!=len(targets) or len(completions)%num_generations:raise ValueError('Incomplete reward groups')
 raw=[-1/math.log2(i+2) for i in range(num_generations)]
 weights=[-r/sum(raw) for r in raw]  # Original negative rank-dependent penalty.
 result=[]
 for start in range(0,len(completions),num_generations):
  c=completions[start:start+num_generations];t=targets[start:start+num_generations]
  rule=[float(x.strip('\n\" ')==y.strip('\n\" ')) for x,y in zip(c,t)]
  hit=[x.strip('\n\"')==y.strip('\n\"') for x,y in zip(c,t)]
  rank=[0. if found else weights[i] for i,found in enumerate(hit)] if any(hit) else [0.]*num_generations
  result.extend(a+b for a,b in zip(rule,rank))
 return result

def group_advantages(rewards,num_generations=16):
 groups=rewards.reshape(-1,num_generations)
 return ((groups-groups.mean(1,keepdim=True))/(groups.std(1,keepdim=True)+1e-4)).reshape(-1)

def grpo_loss(logps,ref_logps,advantages,beta=.04):
 delta=ref_logps-logps
 kl=delta.exp()-delta-1
 loss=-(torch.exp(logps-logps.detach())*advantages[:,None]-beta*kl)
 return loss.mean(1).mean(),kl.mean().detach()

def completion_logps(model,ids,mask,completion_length):
 logits=model(input_ids=ids,attention_mask=mask,logits_to_keep=completion_length+1).logits[:,:-1].float()
 labels=ids[:,-completion_length:]
 return logits.gather(-1,labels.unsqueeze(-1)).squeeze(-1)-torch.logsumexp(logits,dim=-1)
