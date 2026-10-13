"""Compare existing decoder-probe outputs without importing Torch/re-exporting."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]); native=root/'native'
references=json.loads((root/'references.json').read_text())
outputs=json.loads((native/'outputs.json').read_text())
if len(outputs)!=3 or len(references)!=3:
    raise AssertionError("all three prefill/decode steps are required")
report=[]
for index,result in enumerate(outputs):
    for name,expected in references[str(index)].items():
        actual=result[name]
        if actual['shape']!=expected['shape']:
            raise AssertionError((index,name,actual['shape'],expected['shape']))
        a=np.fromfile(native/actual['file'],dtype='<f4')
        b=np.fromfile(root/expected['file'],dtype='<f4')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError((index,name,"nonfinite output"))
        np.testing.assert_allclose(a,b,atol=5e-5,rtol=5e-5,err_msg=f'step{index}/{name}')
        report.append({'step':index,'name':name,'max_abs':float(np.max(np.abs(a-b)))})
    for name in ('out_cache_k0','out_cache_v0'):
        if result[name]['shape'][1]!=(8,9,12)[index]:
            raise AssertionError((index,name,result[name]['shape']))
output={'synthetic':True,'trained_model':False,'native_parity':report}
(root/'native-parity-report.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
