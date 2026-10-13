"""Prepare/compare a saved synthetic tiny decoder, without weights/network.

Usage: python tests/prepare_minicpm_decoder_probe.py FIXTURE_DIR CASE_DIR [PROBE]
The fixture is check_trace.py's tiny 64-wide, four-layer decoder.ts/param/bin.
Torch wrapper references share export logic; the separate HF model comparison
is recorded explicitly, not silently substituted for a failed reference.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from transformers import MiniCPMV4_7Config
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'export'))
from minicpm_v47_export import LanguageOnly
from qwen35_export import Qwen35Decoder, _rename_cache_io
from add_kvcache import add_kvcache_to_param


def main():
    source=Path(sys.argv[1]); root=Path(sys.argv[2]); root.mkdir(parents=True,exist_ok=True)
    cfg=MiniCPMV4_7Config(text_config={'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,'num_hidden_layers':4,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':32,'vocab_size':256,'linear_num_key_heads':2,'linear_num_value_heads':2,'linear_key_head_dim':16,'linear_value_head_dim':16,'linear_conv_kernel_dim':4,'layer_types':['linear_attention']*3+['full_attention'],'partial_rotary_factor':1.0,'rope_parameters':{'rope_type':'default','rope_theta':10000000.0,'partial_rotary_factor':1.0,'mrope_section':[6,5,5]},'tie_word_embeddings':False})
    torch.set_num_threads(2); torch.manual_seed(913)
    model=LanguageOnly(cfg).eval().requires_grad_(False); wrapper=Qwen35Decoder(model).eval()
    traced=torch.jit.load(str(source/'decoder.ts')).eval()
    state=traced.state_dict()
    # TorchScript persists rotary buffers that HF registers non-persistent.
    for name in ('inv_freq','original_inv_freq'):
        key='language_model.rotary_emb.'+name
        if key in state:
            torch.testing.assert_close(state.pop(key),getattr(wrapper.language_model.rotary_emb,name),rtol=0,atol=0)
    wrapper.load_state_dict(state,strict=True)
    print('Loaded exact synthetic checkpoint state',flush=True)
    shutil.copyfile(source/'decoder.ncnn.param',root/'decoder.ncnn.param')
    shutil.copyfile(source/'decoder.ncnn.bin',root/'decoder.ncnn.bin')
    _rename_cache_io(root/'decoder.ncnn.param',3,3)
    add_kvcache_to_param(str(root/'decoder.ncnn.param'),scale=None,backup=False)
    hidden=torch.randn(1,12,64)*.25
    pos=torch.arange(12)[None,None].expand(3,1,-1)
    cos,sin=model.model.language_model.rotary_emb(hidden,pos)
    zero=[torch.zeros(1,96,4) for _ in range(3)]+[torch.zeros(1,2,16,16) for _ in range(3)]
    initial={**{f'cache_conv{i}':{'shape':[4,96],'zero':True} for i in range(3)},
             **{f'cache_gdr{i}':{'shape':[16,16,2],'zero':True} for i in range(3)},
             'cache_k0':None,'cache_v0':None}
    outputs={key:'out_'+key for key in initial}
    case={'initial_states':initial,'state_outputs':outputs,'steps':[]}
    references={}; report={'synthetic':True,'trained_model':False,'hf_reference':[]}
    def save(name,value):
        array=value.detach().cpu().numpy().astype('<f4'); array.tofile(root/(name+'.f32'))
        return {'file':name+'.f32','shape':list(reversed(array.shape))}
    start=0
    for index,end in enumerate((8,9,12)):
        causal=torch.triu(torch.full((end,end),float('-inf')),diagonal=1)
        args=(hidden[:,:end],causal[None,None],cos[:,:end,:16],sin[:,:end,:16],*zero)
        with torch.no_grad():
            ref=wrapper(*args)
            hf=model.model.language_model(inputs_embeds=hidden[:,:end],position_ids=pos[:,:,:end],use_cache=False).last_hidden_state
        np.testing.assert_allclose(ref[0].numpy(),hf.numpy(),atol=2e-5,rtol=2e-5)
        report['hf_reference'].append({'length':end,'max_abs':float((ref[0]-hf).abs().max())})
        if index==0:
            with torch.no_grad(): ts=traced(*args)
            for left,right in zip(ref,ts): np.testing.assert_allclose(left.numpy(),right.numpy(),atol=1e-6,rtol=1e-6)
        inputs={'in0':save(f'{index}_hidden',hidden[0,start:end]),
                'in1':save(f'{index}_mask',causal[start:end]),
                'in2':save(f'{index}_cos',cos[0,start:end,:16]),
                'in3':save(f'{index}_sin',sin[0,start:end,:16])}
        case['steps'].append({'inputs':inputs})
        refs={'out0':save(f'ref_{index}_out0',ref[0][0,start:end])}
        for i in range(3):
            refs[f'out_cache_conv{i}']=save(f'ref_{index}_conv{i}',ref[1+i][0])
            refs[f'out_cache_gdr{i}']=save(f'ref_{index}_gdr{i}',ref[4+i][0])
        references[str(index)]=refs; start=end
    (root/'case.json').write_text(json.dumps(case,indent=2))
    (root/'references.json').write_text(json.dumps(references,indent=2))
    if len(sys.argv)>3:
        destination=root/'native'
        subprocess.run([sys.argv[3],str(root/'decoder.ncnn.param'),str(root/'decoder.ncnn.bin'),str(root/'case.json'),str(destination)],check=True)
        results=json.loads((destination/'outputs.json').read_text()); diffs=[]
        assert len(results)==3, 'all three prefill/decode steps are required'
        for i,result in enumerate(results):
            for name,expected in references[str(i)].items():
                actual=result[name]
                assert actual['shape']==expected['shape'],(i,name,actual,expected)
                a=np.fromfile(destination/actual['file'],dtype='<f4'); b=np.fromfile(root/expected['file'],dtype='<f4')
                assert np.isfinite(a).all() and np.isfinite(b).all(), (i,name,'nonfinite output')
                np.testing.assert_allclose(a,b,atol=5e-5,rtol=5e-5,err_msg=f'step{i}/{name}')
                diffs.append({'step':i,'name':name,'max_abs':float(np.max(np.abs(a-b)))})
            assert result['out_cache_k0']['shape'][1]==(8,9,12)[i]
            assert result['out_cache_v0']['shape'][1]==(8,9,12)[i]
        report['native_parity']=diffs
    (root/'parity-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
