"""Synthetic converter contract tests; these are NOT trained model inference."""
import sys, tempfile, unittest, json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from safetensors.torch import save_file
from transformers import MiniCPMV4_7Config
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'export'))
from minicpm_v47_export import LanguageOnly, load_language_only, finalize_runtime_config
from qwen35_export import Qwen35Decoder, PortableGatedNorm
from qwen35_gdr import torch_chunk_gated_delta_rule
from minicpm_v47_language_export import export_tables

class LanguageExportTests(unittest.TestCase):
 def test_cli_publishes_only_after_finalizers(self):
  import minicpm_v47_export as converter
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); model=root/'model';model.mkdir()
   (model/'config.json').write_text(json.dumps({'model_type':'minicpmv4_7'}))
   for fail in (True,False):
    output=root/('failed' if fail else 'complete')
    def fake_export(source,destination,**kwargs):
     self.assertFalse(output.exists())
     self.assertEqual(kwargs['prefix'],output.name)
     (Path(destination)/'model.json').write_text('{}')
    def finalizer(source,destination):
     self.assertFalse(output.exists())
     if fail: raise RuntimeError('simulated tokenizer failure')
     (Path(destination)/'tokenizer-ready').write_text('ready')
    with patch.object(sys,'argv',['export','-m',str(model),'-o',str(output)]), \
         patch('minicpm_v47_language_export.export_language_components'), \
         patch.object(converter,'export',side_effect=fake_export), \
         patch.object(converter,'finalize_runtime_config'), \
         patch.object(converter,'finalize_tokenizer',side_effect=finalizer):
     if fail:
      with self.assertRaisesRegex(RuntimeError,'simulated'): converter.main()
      self.assertFalse(output.exists())
      self.assertEqual(len(list(root.glob('.failed.staging-*'))),1)
     else:
      converter.main()
      self.assertTrue((output/'tokenizer-ready').is_file())
      self.assertFalse(list(root.glob('.complete.staging-*')))
      with self.assertRaisesRegex(ValueError,'already exists'): converter.main()
 def test_tables_never_reuse_same_shape_stale_weights(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); output=root/'out'
   for value in (1.,2.):
    save_file({'model.language_model.embed_tokens.weight':torch.full((4,2),value),
               'lm_head.weight':torch.full((4,2),value+1)},str(root/'model.safetensors'))
    export_tables(root,output,SimpleNamespace(vocab_size=4,hidden_size=2),'model')
    np.testing.assert_array_equal(np.fromfile(output/'model_embed_token.ncnn.bin',dtype='<f4',offset=4),np.full(8,value,dtype=np.float32))
    np.testing.assert_array_equal(np.fromfile(output/'model_proj_out.ncnn.bin',dtype='<f4',offset=4),np.full(8,value+1,dtype=np.float32))
 def test_runtime_has_no_implicit_bos(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); model=root/'model'; output=root/'output';model.mkdir();output.mkdir()
   config={'tokenizer':{'bos':'<|im_start|>','eos_ids':[7],'additional_special_tokens':['<|im_start|>']},'setting':{'functions':{'type':'tool_call'}}}
   (output/'model.json').write_text(json.dumps(config))
   finalize_runtime_config(model,output)
   result=json.loads((output/'model.json').read_text())
   self.assertEqual(result['tokenizer']['bos'],'')
   self.assertNotIn('functions',result['setting'])
   self.assertEqual(result['tokenizer']['additional_special_tokens'],['<|im_start|>'])
   self.assertEqual(result['tokenizer']['eos_ids'],[7])
   (model/'generation_config.json').write_text(json.dumps({'eos_token_id':[248044,248046]}))
   finalize_runtime_config(model,output)
   self.assertEqual(json.loads((output/'model.json').read_text())['tokenizer']['eos_ids'],[248044,248046])
 def test_stream_load_and_decoder_reference(self):
  torch.manual_seed(4); torch.set_num_threads(2)
  cfg=MiniCPMV4_7Config(text_config={'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,'num_hidden_layers':4,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':32,'vocab_size':256,'linear_num_key_heads':2,'linear_num_value_heads':2,'linear_key_head_dim':16,'linear_value_head_dim':16,'linear_conv_kernel_dim':4,'layer_types':['linear_attention']*3+['full_attention'],'partial_rotary_factor':1.0,'rope_parameters':{'rope_type':'default','rope_theta':10000000.0,'partial_rotary_factor':1.0,'mrope_section':[6,5,5]},'tie_word_embeddings':False})
  original=LanguageOnly(cfg).eval()
  with tempfile.TemporaryDirectory() as td:
   save_file(original.state_dict(),str(Path(td)/'model.safetensors'))
   loaded=load_language_only(td,cfg)
   for n,v in original.named_parameters(): torch.testing.assert_close(v,dict(loaded.named_parameters())[n],atol=0,rtol=0)
   decoder_only=load_language_only(td,cfg,component='decoder')
   self.assertEqual(decoder_only.model.language_model.embed_tokens.weight.shape[0],1)
   self.assertTrue(decoder_only.lm_head.weight.is_meta)
   for n,v in original.model.language_model.layers.named_parameters():
    torch.testing.assert_close(v,dict(decoder_only.model.language_model.layers.named_parameters())[n],atol=0,rtol=0)
   out=Path(td)/'ncnn';export_tables(td,out,cfg.text_config,'tiny')
   import ncnn
   for name,input_data,expected in (
    ('embed_token',np.array([0,7,255,2],dtype=np.int32),original.model.language_model.embed_tokens(torch.tensor([0,7,255,2])).detach().numpy()),
    ('proj_out',np.random.default_rng(2).normal(size=(3,64)).astype(np.float32),None)):
    if expected is None:expected=input_data@original.lm_head.weight.detach().numpy().T
    net=ncnn.Net();net.opt.num_threads=2;net.opt.use_fp16_storage=False;net.opt.use_fp16_packed=False;net.opt.use_bf16_storage=False;net.opt.use_packing_layout=False
    self.assertEqual(net.load_param(str(out/f'tiny_{name}.ncnn.param')),0)
    self.assertEqual(net.load_model(str(out/f'tiny_{name}.ncnn.bin')),0)
    ex=net.create_extractor();owner=np.ascontiguousarray(input_data);mat=ncnn.Mat(owner).clone()
    self.assertEqual(ex.input('in0',mat),0);ret,result=ex.extract('out0');self.assertEqual(ret,0)
    np.testing.assert_allclose(np.array(result).copy(),expected,atol=2e-5,rtol=2e-5)
  for layer in loaded.model.language_model.layers:
   if getattr(layer,'layer_type',getattr(layer,'block_type',None))=='linear_attention':
    layer.linear_attn.chunk_gated_delta_rule=torch_chunk_gated_delta_rule
    layer.linear_attn.recurrent_gated_delta_rule=torch_chunk_gated_delta_rule
    layer.linear_attn.norm=PortableGatedNorm(layer.linear_attn.norm)
  wrapper=Qwen35Decoder(loaded).eval(); lm=loaded.model.language_model
  ids=torch.arange(8)[None]; hidden=lm.embed_tokens(ids)
  pos=torch.arange(8)[None,None].expand(3,1,-1); cos,sin=lm.rotary_emb(hidden,pos)
  mask=torch.triu(torch.full((8,8),float('-inf')),diagonal=1)[None,None]
  states=[torch.zeros(1,96,4) for _ in range(3)]+[torch.zeros(1,2,16,16) for _ in range(3)]
  with torch.no_grad():
   expected=lm(inputs_embeds=hidden,use_cache=False).last_hidden_state
   actual=wrapper(hidden,mask,cos[...,:16],sin[...,:16],*states)[0]
  torch.testing.assert_close(actual,expected,atol=3e-3,rtol=3e-3)
  self.assertFalse(any(x.is_meta for x in loaded.parameters()))
  self.assertFalse(any(x.is_meta for x in loaded.buffers()))

if __name__=='__main__': unittest.main()
