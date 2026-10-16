import ast
import math
from pathlib import Path
import unittest
import torch
from ranking_grpo import ranking_rewards,group_advantages,grpo_loss,completion_logps
from rl_mps import SIDTrie

class RankingGRPOTests(unittest.TestCase):
 def test_rewards_match_original_rl_ranking_branch(self):
  tree=ast.parse((Path(__file__).resolve().parents[1]/'rl.py').read_text())
  train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
  functions=[n for n in train.body if isinstance(n,ast.FunctionDef) and n.name in ['rule_reward','ndcg_rule_reward']]
  groups=4;g=4;prompts=[f'p{i}' for i in range(groups) for _ in range(g)]
  completions=['wrong','ok','wrong','wrong','none']*3+['ok']
  ns=dict(num_generations=g,ndcg_rewards=[-(-1/math.log2(i+2))/sum(-1/math.log2(j+2) for j in range(g)) for i in range(g)],
   prompt2history={f'p{i}':f'h{i}' for i in range(groups)},history2target={f'h{i}':'ok\n' for i in range(groups)})
  exec(compile(ast.Module(body=functions,type_ignores=[]),'<original rewards>','exec'),ns)
  expected=[a+b for a,b in zip(ns['rule_reward'](prompts,completions),ns['ndcg_rule_reward'](prompts,completions))]
  self.assertEqual(ranking_rewards(completions,['ok\n']*len(completions),g),expected)
 def test_all_miss_group_has_zero_reward_and_advantage(self):
  r=torch.tensor(ranking_rewards(['x']*16,['target']*16))
  torch.testing.assert_close(r,torch.zeros_like(r));torch.testing.assert_close(group_advantages(r),r)
 def test_microbatch_loss_and_gradients_match_full_group(self):
  torch.manual_seed(0)
  logps=torch.randn(16,5,requires_grad=True);ref=torch.randn(16,5);adv=group_advantages(torch.arange(16).float())
  loss,kl=grpo_loss(logps,ref,adv);loss.backward();expected=logps.grad.clone()
  logps.grad=None
  for i in range(0,16,4):
   part,_=grpo_loss(logps[i:i+4],ref[i:i+4],adv[i:i+4]);(part*4/16).backward()
  torch.testing.assert_close(expected,logps.grad)
  delta=ref-logps.detach()
  original=-((logps-logps.detach()).exp()*adv[:,None]-.04*(delta.exp()-delta-1))
  torch.testing.assert_close(loss.detach(),original.mean(1).mean())
 def test_selective_completion_logps_match_full_logits(self):
  from transformers import Qwen2Config,Qwen2ForCausalLM
  model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,attention_dropout=0.))
  model.eval();ids=torch.tensor([[1,2,3,4,5,6],[3,4,5,6,7,8]]);mask=torch.ones_like(ids)
  logits=model(input_ids=ids,attention_mask=mask).logits[:,-4:-1].float()
  expected=logits.log_softmax(-1).gather(-1,ids[:,-3:].unsqueeze(-1)).squeeze(-1)
  torch.testing.assert_close(completion_logps(model,ids,mask,3),expected)
 def test_cpu_constrained_beam_generation_returns_legal_unique_items(self):
  from transformers import Qwen2Config,Qwen2ForCausalLM
  model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,attention_dropout=0.,eos_token_id=31,pad_token_id=31))
  legal=[[4,6,8,30,31],[4,7,9,30,31],[5,6,10,30,31],[5,7,11,30,31]];trie=SIDTrie(legal)
  prompt=torch.tensor([[1,2,3]])
  result=model.generate(prompt,attention_mask=torch.ones_like(prompt),num_beams=4,num_return_sequences=4,max_new_tokens=5,
   eos_token_id=31,pad_token_id=31,prefix_allowed_tokens_fn=lambda _,tokens:trie.allowed(tokens[3:].tolist()),do_sample=False)
  self.assertEqual({tuple(x) for x in result[:,3:].tolist()},{tuple(x) for x in legal})

if __name__=='__main__':unittest.main()
