"""Actual Transformers reference parity for independent C++ MiniCPM helpers.

No checkpoint tensors or language/vision modules are allocated. Uses the real
MiniCPMV4_7Model positioning methods and Qwen3_5TextRotaryEmbedding from installed
Transformers; config is read from the requested checkpoint directory.
"""
from minicpm_test_build import probe_path
import argparse
import pathlib
import subprocess
import tempfile

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7Model
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from transformers.vision_utils import get_vision_nearest_position_ids


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    # Only pure positioning methods need config. Bypass constructor intentionally
    # to avoid allocating either checkpoint-sized or random surrogate networks.
    model = MiniCPMV4_7Model.__new__(MiniCPMV4_7Model)
    torch.nn.Module.__init__(model)
    model.config = cfg
    model.rope_deltas = None
    rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config)
    params = cfg.text_config.rope_parameters
    dim = int(cfg.text_config.head_dim * params['partial_rotary_factor'])
    markers = [cfg.image_token_id, cfg.image_start_id, cfg.image_end_id,
               cfg.slice_start_id, cfg.slice_end_id, cfg.newline_id]
    side = cfg.vision_config.image_size // cfg.vision_config.patch_size
    root = pathlib.Path(__file__).resolve().parents[1]
    cases = []
    for mode in ('16x', '4x'):
        merge = 4 if mode == '16x' else 2
        for h,w in ((32,32), (36,28), (28,36), (4,4), (8,12), (12,8), (68,72)):
            for base in (0,17,513):
                for newlines in (0,1,3):
                    for suffix in ([], [41,42]):
                        ids = [31,32] + [cfg.image_start_id] + [cfg.image_token_id]*(h*w//merge**2)
                        ids += [cfg.image_end_id]+[cfg.newline_id]*newlines+suffix
                        cases.append((mode,h,w,base,ids))
    records = []
    for mode,h,w,base,ids in cases:
        fields = [mode,h,w,side,base,dim,params['rope_theta'],*markers,*params['mrope_section'],len(ids),*ids]
        records.append(' '.join(map(str,fields)))
    with tempfile.TemporaryDirectory() as td:
        probe = pathlib.Path(td)/'probe'
        probe = probe_path('minicpm_vision_reference_probe', 'executable')
        result = subprocess.run([str(probe)],input='\n'.join(records)+'\n',capture_output=True,text=True,check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == 5*len(cases)
    max_rope_error = 0.0
    for i,(mode,h,w,base,ids) in enumerate(cases):
        label = (mode,h,w,base,len(ids))
        delta,next_pos = map(int,lines[i*5].split())
        actual = np.fromstring(lines[i*5+1],sep=' ',dtype=np.int64).reshape(-1,3)
        nearest = np.fromstring(lines[i*5+2],sep=' ',dtype=np.int64)
        cc = np.fromstring(lines[i*5+3],sep=' ',dtype=np.float32).reshape(-1,dim//2)
        ss = np.fromstring(lines[i*5+4],sep=' ',dtype=np.float32).reshape(-1,dim//2)
        # A separately cached system prefix is represented by `base` plain text
        # tokens in the full HF reference. Check only the current suffix region.
        full_ids = torch.tensor([[77]*base+ids],dtype=torch.long)
        mm = (full_ids == cfg.image_token_id).int()
        ref_pos,ref_delta = model.get_rope_index(full_ids,mm_token_type_ids=mm,
                           target_sizes=torch.tensor([[h,w]]),downsample_mode=mode)
        expected = ref_pos[:,0,base:].T.numpy()
        np.testing.assert_array_equal(actual,expected,err_msg=str(label))
        # Helper delta is relative to suffix length. Cached text contributes base
        # to that delta; the canonical HF full-sequence delta excludes it.
        assert delta-base == ref_delta.item(), (label,delta,ref_delta)
        assert next_pos == full_ids.shape[-1]+ref_delta.item(), (label,next_pos,ref_delta)
        nearest_ref = get_vision_nearest_position_ids(torch.tensor([[h,w]]),side).numpy()
        np.testing.assert_array_equal(nearest,nearest_ref,err_msg=str(label))
        with torch.no_grad():
            c,s = rotary(torch.zeros(1,len(ids),1),ref_pos[:,:,base:])
        for out,ref in ((cc,c[0,:,:dim//2].numpy()),(ss,s[0,:,:dim//2].numpy())):
            max_rope_error = max(max_rope_error,float(np.max(np.abs(out-ref))))
            np.testing.assert_allclose(out,ref,atol=7e-5,rtol=1e-5,err_msg=str(label))
        # Official cached generation method must continue from next_position.
        model.rope_deltas = ref_delta
        class CacheLength:
            def get_seq_length(self): return full_ids.shape[-1]
        decoded = model.compute_3d_position_ids(None,inputs_embeds=torch.zeros(1,3,1),
                                                past_key_values=CacheLength())
        np.testing.assert_array_equal(decoded[:,0].numpy(),
                                      np.tile(np.arange(next_pos,next_pos+3),(3,1)),err_msg=str(label))
    print(f'PASS: {len(cases)} real HF canvas/nearest/RoPE/cached-decode comparisons; max RoPE abs error={max_rope_error:.9g}')
    print('No model weights loaded; this establishes helper numerical parity only.')

if __name__ == '__main__':
    main()
