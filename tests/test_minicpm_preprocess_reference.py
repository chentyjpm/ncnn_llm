"""Measure complete C++ preprocessing against actual HF Torchvision/PIL backends."""
from minicpm_test_build import probe_path
import argparse
import ctypes
import io
import json
import pathlib
import subprocess
import tempfile
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor
from transformers.models.minicpmv4_6.image_processing_minicpmv4_6 import MiniCPMV4_6ImageProcessor
from transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6 import MiniCPMV4_6ImageProcessorPil


def main():
    torch.set_num_threads(1)
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir',required=True)
    ap.add_argument('--output')
    ap.add_argument('--fixture', help='Optional PNG to save actual processor tensors for')
    ap.add_argument('--fixture-output-prefix', default='processor')
    args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1]
    processor=AutoImageProcessor.from_pretrained(args.model_dir,local_files_only=True)
    print('Auto processor:',type(processor).__name__,flush=True)
    pil=MiniCPMV4_6ImageProcessorPil.from_pretrained(args.model_dir,local_files_only=True)
    processors={'auto':processor,'pil':pil}
    if not isinstance(processor,MiniCPMV4_6ImageProcessor):
        processors['torchvision']=MiniCPMV4_6ImageProcessor.from_pretrained(args.model_dir,local_files_only=True)
    if args.fixture:
        image=Image.open(args.fixture).convert('RGB')
        for name,p in processors.items():
            out=p(images=image,max_slice_nums=1,use_image_id=False,return_tensors='pt')
            prefix=args.fixture_output_prefix+('-pil' if name=='pil' else '')
            np.save(prefix+'-pixels.npy',out['pixel_values'].float().cpu().numpy())
            np.save(prefix+'-target-sizes.npy',out['target_sizes'].cpu().numpy())
            print('Saved actual processor tensors:',prefix,flush=True)
    rng=np.random.default_rng(4710)
    dims=[(448,448),(100,100),(300,400),(400,300),(900,1200),(1200,900),(53,511),(511,53),(1,1),(56,56)]
    results=[]
    with tempfile.TemporaryDirectory() as td:
        so=pathlib.Path(td)/'probe.so'
        so = probe_path('minicpm_preprocess_probe', 'library')
        run=ctypes.CDLL(str(so)).minicpm_preprocess_probe
        up=ctypes.POINTER(ctypes.c_ubyte); fp=ctypes.POINTER(ctypes.c_float); ip=ctypes.POINTER(ctypes.c_int)
        run.argtypes=[up,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,fp,ctypes.c_int,ip]
        for h,w in dims:
            for pattern in ('random','checkerboard'):
                rgb=rng.integers(0,256,(h,w,3),dtype=np.uint8)
                if pattern=='checkerboard': rgb=np.repeat(((np.indices((h,w)).sum(0)%2)*255).astype(np.uint8)[:,:,None],3,axis=2)
                # Exercise real lossless RGB image decoding, not an already normalized tensor.
                buf=io.BytesIO(); Image.fromarray(rgb).save(buf,format='PNG'); buf.seek(0)
                image=Image.open(buf).convert('RGB')
                actual=None
                for name,p in processors.items():
                    out=p(images=image,max_slice_nums=1,use_image_id=False,return_tensors='pt')
                    ref=out['pixel_values'].float().cpu().numpy().reshape(-1)
                    if actual is None:
                        actual=np.empty(ref.size,dtype=np.float32); shape=np.zeros(2,dtype=np.int32)
                        size=run(rgb.ctypes.data_as(up),h,w,p.patch_size,p.scale_resolution,
                                 actual.ctypes.data_as(fp),actual.size,shape.ctypes.data_as(ip))
                        assert size==ref.size,(size,ref.size)
                    np.testing.assert_array_equal(shape,out['target_sizes'][0].cpu().numpy())
                    diff=np.abs(actual-ref)
                    # Quantify resizing separately from float normalization rounding.
                    actual_u8=np.rint((actual+1)*127.5).astype(np.uint8)
                    ref_u8=np.rint((ref+1)*127.5).astype(np.uint8)
                    item={'backend':name,'class':type(p).__name__,'height':h,'width':w,'pattern':pattern,
                          'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
                          'different_pixels':int(np.count_nonzero(actual_u8!=ref_u8)),
                          'max_byte_difference':int(np.max(np.abs(actual_u8.astype(int)-ref_u8.astype(int))))}
                    results.append(item)
                    print(json.dumps(item),flush=True)
    summary={name:{'max_abs':max(r['max_abs'] for r in results if r['backend']==name),
                   'max_byte_difference':max(r['max_byte_difference'] for r in results if r['backend']==name),
                   'different_pixels':sum(r['different_pixels'] for r in results if r['backend']==name)}
             for name in processors}
    print('SUMMARY',json.dumps(summary),flush=True)
    if args.output: pathlib.Path(args.output).write_text(json.dumps({'results':results,'summary':summary},indent=2))
    # PIL backend is the explicitly selected reference for this helper. Report
    # default Torchvision discrepancies rather than silently treating them as parity.
    assert summary['pil']['different_pixels']==0,summary
    assert summary['pil']['max_abs']<3e-7,summary

if __name__=='__main__': main()
