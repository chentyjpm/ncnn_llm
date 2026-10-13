"""Byte-exact comparison of C++ thumbnail bicubic resize against Pillow.
No model weights, Torch, or ncnn runtime needed.
"""
import ctypes
import pathlib
import subprocess
import tempfile
import numpy as np
import PIL
from PIL import Image


def main():
    root=pathlib.Path(__file__).resolve().parents[1]
    rng=np.random.default_rng(470123)
    dimensions=[(1,1,3,5),(1,7,9,3),(7,1,3,9),(2,2,1,1),(3,7,3,7),
                (200,300,448,672),(900,1200,392,504),(1200,900,504,392),
                (1000,1000,448,448),(56,56,448,448),(448,448,56,56),
                (60,400,700,28),(400,60,28,700),(32,40,32,10),(32,40,8,40)]
    dimensions += [tuple(int(x) for x in rng.integers(1,200,size=4)) for _ in range(200)]
    with tempfile.TemporaryDirectory() as td:
        so=pathlib.Path(td)/'probe.so'
        subprocess.run(['g++','-std=c++11','-O2','-Wall','-Wextra','-Werror','-pedantic',
                        '-shared','-fPIC','-I'+str(root/'src'),str(root/'tests/minicpm_resize_probe.cpp'),
                        str(root/'src/utils/minicpm_resize.cpp'),'-o',str(so)],check=True)
        lib=ctypes.CDLL(str(so))
        pointer=ctypes.POINTER(ctypes.c_ubyte)
        run=lib.minicpm_resize_probe
        run.argtypes=[pointer,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,pointer]
        run.restype=ctypes.c_int
        count=0
        for h,w,th,tw in dimensions:
            images=[rng.integers(0,256,size=(h,w,3),dtype=np.uint8)]
            # Sharp checkerboard and impulse exercise negative lobes/clipping.
            checker=((np.indices((h,w)).sum(axis=0)%2)*255).astype(np.uint8)
            images.append(np.repeat(checker[:,:,None],3,axis=2))
            impulse=np.zeros((h,w,3),dtype=np.uint8); impulse[h//2,w//2]=[255,127,61]
            images.append(impulse)
            for original in images:
                expected=np.asarray(Image.fromarray(original).resize((tw,th),Image.Resampling.BICUBIC))
                for bgr in (False,True):
                    src=np.ascontiguousarray(original[:,:,::-1] if bgr else original)
                    actual=np.empty((th,tw,3),dtype=np.uint8)
                    status=run(src.ctypes.data_as(pointer),h,w,th,tw,bgr,actual.ctypes.data_as(pointer))
                    assert status==0
                    np.testing.assert_array_equal(actual,expected,err_msg=str((h,w,th,tw,bgr)))
                    count+=1
        dummy=np.zeros(3,dtype=np.uint8)
        assert run(dummy.ctypes.data_as(pointer),0,1,1,1,0,dummy.ctypes.data_as(pointer))==-1
        assert run(dummy.ctypes.data_as(pointer),1,1,0,1,0,dummy.ctypes.data_as(pointer))==-1
    print(f'PASS: {count} byte-exact RGB/BGR bicubic comparisons against Pillow {PIL.__version__}; invalid dimensions rejected')

if __name__=='__main__': main()
