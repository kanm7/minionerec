import unittest
from sft_mps import collate_sft

class SFTCollatorTests(unittest.TestCase):
 def test_prompt_mask_and_response_labels_survive_padding(self):
  rows=[dict(input_ids=[10,11,12],attention_mask=[1,1,1],labels=[-100,-100,12]),
        dict(input_ids=[20,21,22,23,24],attention_mask=[1]*5,labels=[-100,-100,22,23,24])]
  batch=collate_sft(rows,99)
  self.assertEqual(batch['labels'].tolist(),[[-100,-100,12,-100,-100],[-100,-100,22,23,24]])
  self.assertEqual(batch['input_ids'][0].tolist(),[10,11,12,99,99])
  self.assertEqual(batch['attention_mask'][0].tolist(),[1,1,1,0,0])
  self.assertEqual(int((batch['labels'][:,1:]!=-100).sum()),4)
if __name__=='__main__':unittest.main()

class ResponseLossEquivalenceTests(unittest.TestCase):
 def test_matches_full_logits_loss_and_gradients(self):
  import torch
  from transformers import Qwen2Config,Qwen2ForCausalLM
  from sft_mps import response_loss
  torch.manual_seed(42)
  model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,attention_dropout=0.,tie_word_embeddings=True))
  model.eval()
  batch=collate_sft([dict(input_ids=[1,2,3,4,5],attention_mask=[1]*5,labels=[-100,-100,-100,4,5]),dict(input_ids=[6,7,8],attention_mask=[1]*3,labels=[-100,7,8])],0)
  original=model(**batch).loss;original.backward()
  grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
  model.zero_grad(set_to_none=True)
  optimized=response_loss(model,batch);optimized.backward()
  torch.testing.assert_close(original,optimized,rtol=1e-6,atol=1e-6)
  for n,p in model.named_parameters():
   if n in grads:torch.testing.assert_close(grads[n],p.grad,rtol=2e-5,atol=2e-6)
