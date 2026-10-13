"""Actual trained single-thumbnail HF-to-ncnn visual-feature parity (CPU FP32)."""
import argparse, gc, json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model_dir", type=Path)
parser.add_argument("image", type=Path, help="448x448 RGB photo; preprocessing is verified separately")
parser.add_argument("output_dir", type=Path)
parser.add_argument("converted_dir", type=Path)
parser.add_argument("--prefix", default="")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--atol", type=float, default=1e-3)
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--min-cosine", type=float, default=.999999)
args = parser.parse_args()
ROOT, OUT, CONVERTED, PREFIX = args.model_dir, args.output_dir, args.converted_dir, args.prefix
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from minicpm_v47_vision_export import export_minicpm_v47_vision, window_indices, nearest_position_ids

OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(args.threads)
provenance_path = ROOT/'download-manifest.json'
provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
entry = next((f for f in provenance.get('files', []) if f['name']=='model.safetensors'), None)
assert (ROOT/'model.safetensors').exists(), 'A complete local safetensors checkpoint is required'
if entry is not None:
    assert (ROOT/'model.safetensors').stat().st_size == entry['bytes'], 'Verified checkpoint size mismatch'
else:
    entry = {'sha256': None}

image = np.asarray(Image.open(args.image).convert('RGB'))
assert image.shape == (448,448,3)
pixels = image.transpose(2,0,1).astype(np.float32)/127.5 - 1
# Standard RGB CHW produces the same patch row order as processor NaViT packing.
np.save(OUT/'pixels_chw.npy', pixels)
targets=torch.tensor([[32,32]], dtype=torch.int32)
from transformers.models.minicpmv4_7.configuration_minicpmv4_7 import MiniCPMV4_7Config
from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7Model
cfg=MiniCPMV4_7Config.from_pretrained(ROOT,local_files_only=True)
cfg._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
print('Constructing meta reference; language tensors remain meta',flush=True)
with torch.device('meta'):
    model=MiniCPMV4_7Model(cfg)
with safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu') as sf:
    count=0
    for key in sf.keys():
        if not key.startswith(('model.vision_tower.','model.merger.')):
            continue
        relative=key.removeprefix('model.')
        parent,name=relative.rsplit('.',1)
        module=model.get_submodule(parent)
        assert name in module._parameters,(key,'not a parameter')
        module._parameters[name]=torch.nn.Parameter(sf.get_tensor(key).float(),requires_grad=False)
        count+=1
print('Loaded vision parameters',count,flush=True)
assert all(not p.is_meta for p in model.vision_tower.parameters())
assert all(not p.is_meta for p in model.merger.parameters())
assert all(p.is_meta for p in model.language_model.parameters())
assert not any(b.is_meta for b in model.vision_tower.buffers()),'Unexpected vision meta buffers'
model.eval()
def hook(name):
    def save(_module,_inputs,result):
        if isinstance(result,tuple):result=result[0]
        np.save(OUT/(name+'.npy'),result.detach().float().cpu().numpy().squeeze(0))
    return save
handles=[]
for name,mod in [('hf_embedding',model.vision_tower.embeddings),('hf_pre',model.vision_tower.encoder.layers[6]),('hf_window',model.vision_tower.vit_merger),('hf_post',model.vision_tower.post_layernorm)]:
    handles.append(mod.register_forward_hook(hook(name)))
start=time.time()
with torch.no_grad():
    features=model.get_image_features(torch.from_numpy(pixels).unsqueeze(0),targets,downsample_mode='16x').pooler_output[0]
    np.save(OUT/'hf_features.npy',features.numpy())
print('HF vision seconds',time.time()-start,'shape',tuple(features.shape),flush=True)
for h in handles:h.remove()
del handles, model, features, module, mod
# Ensure references from construction/loading loops don't retain large tensors.
gc.collect()
print('HF reference released; exporting streamed vision components',flush=True)
manifest=export_minicpm_v47_vision(ROOT,CONVERTED,prefix=PREFIX)
import ncnn
reports={}
def compare(name,actual):
    expected=np.load(OUT/('hf_'+name+'.npy'))
    np.save(OUT/('ncnn_'+name+'.npy'),actual)
    assert actual.shape==expected.shape,(name,actual.shape,expected.shape)
    a,b=actual.astype(np.float64).ravel(),expected.astype(np.float64).ravel()
    delta=np.abs(a-b)
    reports[name]={'shape':list(actual.shape),'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),'cosine_similarity':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))}
    print(name,reports[name],flush=True)
    np.testing.assert_allclose(actual, expected, atol=args.atol, rtol=args.rtol)
    assert reports[name]['cosine_similarity'] > args.min_cosine, reports[name]
def load(name):
    net=ncnn.Net();net.opt.num_threads=args.threads;net.opt.use_vulkan_compute=False
    net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False
    net.opt.use_bf16_storage=False;net.opt.use_packing_layout=False
    item=manifest['components'][name]
    assert net.load_param(str(CONVERTED/item['param']))==0
    assert net.load_model(str(CONVERTED/item['bin']))==0
    return net
def run_net(net,x):
    ex=net.create_extractor()
    owned_input = np.ascontiguousarray(x, dtype=np.float32)
    input_mat = ncnn.Mat(owned_input).clone()
    assert ex.input('in0', input_mat)==0
    ret,out=ex.extract('out0');assert ret==0
    return np.array(out).copy()
def run(name,x):
    print('Running',name,flush=True)
    net=load(name);y=run_net(net,x);del net;gc.collect();return y
x=run('patch_embed',pixels)
pos=np.fromfile(CONVERTED/manifest['position_embeddings']['file'],dtype=np.float32).reshape(4900,1152)
x += pos[nearest_position_ids(32,32)];del pos
compare('embedding',x)
x=run('encoder_pre',x);compare('pre',x)
x=x[window_indices(32,32)].reshape(-1,4,1152)
net=load('window_attn')
y=np.stack([run_net(net,window) for window in x]);del net,x;gc.collect()
x=run('window_mlp',y.reshape(-1,4608))+y.mean(axis=1);del y
compare('window',x)
x=run('encoder_post',x);compare('post',x)
x=x[window_indices(16,16)].reshape(-1,4608)
x=run('projector',x);compare('features',x)
reports['metadata']={'checkpoint_sha256':entry['sha256'],'revision':provenance.get('revision'),'ncnn_version':getattr(ncnn,'__version__','unknown'),'image':str(args.image),'dtype':'float32','pixel_normalization':'RGB/127.5-1','hf_vision_only':True,'acceptance':{'atol':args.atol,'rtol':args.rtol,'min_cosine':args.min_cosine},'input_buffer_retained':True}
(OUT/'parity.json').write_text(json.dumps(reports,indent=2)+'\n')
print('PASS trained visual feature absolute/relative and cosine checks',flush=True)
