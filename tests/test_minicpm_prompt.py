"""Compare the C++ MiniCPM prompt with the checkpoint's Jinja and token IDs.
No torch/transformers imports or weights. Usage: python TEST MODEL_DIR PROBE
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path
from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer

model, probe = Path(sys.argv[1]), sys.argv[2]
def fail(message):
    raise ValueError(message)
env = Environment()
env.globals['raise_exception'] = fail
template = env.from_string(json.loads((model/'tokenizer_config.json').read_text())['chat_template'])
tokenizer = Tokenizer.from_file(str(model/'tokenizer.json'))
cases = []
for thinking, generation, system, history in itertools.product((False,True),repeat=4):
    messages = ([{'role':'system','content':'  You are a helpful assistant.\n'}] if system else [])
    if history:
        messages += [{'role':'user','content':' First question? '},
                     {'role':'assistant','content':' First answer. ','reasoning_content':' secret past thought '}]
    messages += [{'role':'user','content':' <image>\nDescribe this image in one short sentence.\n'}]
    cases.append(dict(messages=messages,enable_thinking=thinking,add_generation_prompt=generation))
for content, reasoning in [('Answer.', ' final thought '), ('<think>\nthought\n</think>\n\nAnswer.', None), ('Answer.',None)]:
    for thinking in (False,True):
        assistant = {'role':'assistant','content':content}
        if reasoning is not None: assistant['reasoning_content']=reasoning
        cases.append(dict(messages=[{'role':'user','content':'Question'},assistant], enable_thinking=thinking,add_generation_prompt=False))
for content in ['a  b', 'a\n\nb', '中文问题：图片里有什么？', '1234567890', ' a\t b ']:
    cases.append(dict(messages=[{'role':'user','content':content}],enable_thinking=False,add_generation_prompt=True))
process=subprocess.run([probe],input=''.join(json.dumps(c)+'\n' for c in cases),text=True,capture_output=True,check=True)
results=[json.loads(line) for line in process.stdout.splitlines()]
assert len(results)==len(cases)
for i,(case,actual) in enumerate(zip(cases,results)):
    expected=template.render(**case)
    assert actual.get('text')==expected,(i,actual,repr(expected))
    a=tokenizer.encode(actual['text'],add_special_tokens=False).ids
    b=tokenizer.encode(expected,add_special_tokens=False).ids
    assert a==b,(i,a,b)
    # The checkpoint's ByteLevel postprocessor adds no BOS, despite bos_token metadata.
    assert tokenizer.encode(expected,add_special_tokens=True).ids==b
print(json.dumps({'passed':len(cases),'exact_strings':True,'official_tokenizer_ids_for_rendered_text':True,'automatic_bos':False}))
