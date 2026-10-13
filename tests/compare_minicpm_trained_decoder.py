"""Compare a trained native decoder probe to a saved independent HF target.
Usage: python TEST FIXTURE_DIR NATIVE_OUTPUT_DIR
This checks decoder hidden states only, not complete image-question inference.
"""
import json
from pathlib import Path
import sys
import numpy as np

root, output = map(Path, sys.argv[1:3])
report = json.loads((output/'outputs.json').read_text())
assert len(report)==1, 'expected one trained eight-token prefill'
spec=report[0]['out0']
assert spec['shape']==[1024,8], spec
actual=np.fromfile(output/spec['file'],dtype='<f4').astype(np.float64)
expected=np.fromfile(root/'target.f32',dtype='<f4').astype(np.float64)
assert actual.shape==expected.shape==(8192,)
assert np.isfinite(actual).all() and np.isfinite(expected).all()
diff=np.abs(actual-expected)
cosine=float(np.dot(actual,expected)/(np.linalg.norm(actual)*np.linalg.norm(expected)))
metrics={'trained_decoder':True,'full_vqa':False,'shape':spec['shape'],
         'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'cosine':cosine,
         'relative_l2':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
         'atol':5e-4,'rtol':5e-4,
         'passed':bool(np.allclose(actual,expected,atol=5e-4,rtol=5e-4) and cosine>=0.99999)}
(output/'trained-parity.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
if not metrics['passed']: raise SystemExit(1)
