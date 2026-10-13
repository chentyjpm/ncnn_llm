"""Actual native BPE ID parity with checkpoint tokenizers backend, no Torch."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'export'))
from minicpm_v47_tokenizer import export_minicpm_v47_tokenizer,validate_pipeline


def cases(data):
    texts=['#include <stdio.h>\n#define FOO 42\n### Header\n########',r'\r\n\t',r'hello\rworld',r'\r',r'\n',r'\t','','a  b','a   b',' a b ','a\t\tb','a\r\n  b',' \n\t \r\n  ','\n\r\n',
           "I'm I'M i'd I'd I've I’M 'ſ 'S 't 're 've 'm 'll 'd",'1234567890 １２３ ①Ⅱ²',
           '你好，世界！中文 English123混合','🙂hello 👨‍👩‍👧‍👦!  🚀🚀abc','e\u0301 A\u030a \u212b',
           '각각한글','\u0315\u0300A\u0301\u0327','किताब हिन्दी বাংলা தமிழ்',
           'שלום עולם العربية فارسی','\x00A\x01\x1cB\x85C\u00a0D\u2003E\u2028F\u2029G\u3000H',
           'def foo(x=123):\n    return "a  b"\n\n# 中文注释\n','!!!abc !abc .\r\n ',
           '<|im_start|>user\n<image><|image_pad|></image>\nDescribe.<|im_end|>\n',
           '<think>\n\n</think>\n\n<tool_call>{"a":1}</tool_call>',
           '\U00011938\U0001e4ec\U0001e4ed\u0300']
    for item in data['added_tokens']:
        texts.extend([item['content'],'a '+item['content']+' b','e\u0301'+item['content']+'A\u030a'])
    rng=random.Random(47)
    whitespace=' \t\n\r\f\v\x85\u00a0\u1680\u2000\u200b\u2028\u2029\u202f\u205f\u3000'
    alphabet=list("abcXYZstrEVmLD'!?.,{}[]()/\\0123456789")+list(whitespace)+list('你好世界中文éÅßſİı각🙂🚀')
    alphabet+=list('\u0300\u0301\u0308\u0315\u0327\u0345\u093c\u05b0\u05b1')
    for _ in range(3000): texts.append(''.join(rng.choices(alphabet,k=rng.randrange(0,120))))
    # Random scalars cover classification and byte-level fallback independently
    # of ordinary language fixtures. Exclude surrogate code points only.
    for _ in range(500):
        row=[]
        for _ in range(rng.randrange(1,50)):
            cp=rng.randrange(0x110000)
            if not 0xd800<=cp<=0xdfff: row.append(chr(cp))
        texts.append(''.join(row))
    # Every non-Hangul decomposition and every class-table scalar is sampled
    # in surrounding text, exercising C++ NFC rather than only Python tables.
    return texts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True)
    ap.add_argument('--artifact-dir');ap.add_argument('--report')
    args=ap.parse_args()
    raw=Path(args.model_dir)/'tokenizer.json';data=json.loads(raw.read_text());validate_pipeline(data)
    for field, value in [('decoder', None), ('post_processor', None),
                         ('truncation', {}), ('padding', {})]:
        unsupported=copy.deepcopy(data);unsupported[field]=value
        try: validate_pipeline(unsupported)
        except ValueError: pass
        else: raise AssertionError('accepted unsupported '+field)
    reference=Tokenizer.from_file(str(raw))
    texts=cases(data)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); artifact_dir=Path(args.artifact_dir) if args.artifact_dir else td
        artifact=artifact_dir/'minicpm_v47_tokenizer.txt'
        if not artifact.exists(): export_minicpm_v47_tokenizer(args.model_dir,artifact_dir)
        for line in artifact.read_text().splitlines():
            fields=line.split()
            if fields and fields[0] in ('D','C'):
                cp=int(fields[1]);texts.append('a'+chr(cp)+'\u0301\u0327 b')
        by_id={v:k for k,v in data['model']['vocab'].items()}
        by_id.update({v['id']:v['content'] for v in data['added_tokens']})
        assert sorted(by_id)==list(range(len(by_id)))
        (td/'vocab.txt').write_text(''.join(by_id[i].replace('\n','\\n').replace('\r','\\r')+'\n' for i in range(len(by_id))))
        (td/'merges.txt').write_text(''.join((' '.join(m) if isinstance(m,list) else m)+'\n' for m in data['model']['merges']))
        probe=td/'probe'
        subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT/'src'),
                        str(ROOT/'tests/minicpm_tokenizer_probe.cpp'),str(ROOT/'src/utils/tokenizer/bpe_tokenizer.cpp'),
                        '-o',str(probe)],check=True)
        payload='\n'.join(t.encode().hex() for t in texts)+'\n'
        process=subprocess.run([str(probe),str(td/'vocab.txt'),str(td/'merges.txt'),str(artifact)],
                               input=payload,text=True,capture_output=True,check=True)
        rows=process.stdout.splitlines();assert len(rows)==len(texts),(len(rows),len(texts),process.stderr)
        failures=[]
        for i,(text,row) in enumerate(zip(texts,rows)):
            if row.startswith('ERROR:'): failures.append({'index':i,'text':text,'error':row});continue
            raw_ids,decoded,skipped=row.split('|')
            actual=[int(x) for x in raw_ids.split(',')] if raw_ids else []
            expected=reference.encode(text,add_special_tokens=False).ids
            if actual!=expected:
                failures.append({'index':i,'text':text,'actual':actual,'expected':expected})
            # decode byte sequences may be incomplete only for synthetic ID lists;
            # complete encoded inputs here should match backend normalization.
            if bytes.fromhex(decoded).decode('utf-8',errors='replace')!=reference.decode(expected,skip_special_tokens=False):
                failures.append({'index':i,'text':text,'decode_mismatch':True})
            if bytes.fromhex(skipped).decode('utf-8',errors='replace')!=reference.decode(expected,skip_special_tokens=True):
                failures.append({'index':i,'text':text,'skip_special_mismatch':True})
        # Fail closed for invalid UTF-8 and unsupported regex artifacts.
        bad=subprocess.run([str(probe),str(td/'vocab.txt'),str(td/'merges.txt'),str(artifact)],input='80\nc080\neda080\nf4908080\ne282\n',text=True,capture_output=True,check=True)
        assert len(bad.stdout.splitlines())==5 and all(line.startswith('ERROR:') for line in bad.stdout.splitlines())
        original=artifact.read_text()
        records=original.splitlines()
        mutations={'regex':original.replace("(?i:'s", "(?i:'x",1),
                   'missing_end':original.replace('END\n',''),
                   'source_hex':original.replace(next(r for r in records if r.startswith('source ')), 'source '+'z'*64),
                   'decomposition_count':original.replace(next(r for r in records if r.startswith('D ')), 'D 65 4294967295'),
                   'added_vocab_mismatch':original.replace(next(r for r in records if r.startswith('A ')), 'A 0 1 616263'),
                   'excess_fields':original.replace('backend ', 'backend extra ',1)}
        for kind in ('R','D','C','P','A'):
            record=next(r for r in records if r.startswith(kind+' '))
            mutations['duplicate_'+kind]=original.replace(record,record+'\n'+record,1)
        for name, content in mutations.items():
            invalid=td/(name+'.txt');invalid.write_text(content)
            rejected=subprocess.run([str(probe),str(td/'vocab.txt'),str(td/'merges.txt'),str(invalid)],input='',text=True,capture_output=True)
            assert rejected.returncode!=0, 'accepted invalid artifact '+name
    report={'case_count':len(texts),'failure_count':len(failures),'failures':failures,
            'reference':'actual tokenizers.Tokenizer.from_file(checkpoint/tokenizer.json)',
            'invalid_utf8_rejected':True,'unsupported_regex_rejected':True,
            'invalid_artifact_cases':list(mutations),'unsupported_pipeline_cases':4}
    if args.report: Path(args.report).write_text(json.dumps(report,ensure_ascii=True,indent=2))
    print(json.dumps({**report,'failures':failures[:5]},ensure_ascii=True,indent=2),flush=True)
    assert not failures,'native tokenizer differs from checkpoint backend'

if __name__=='__main__':main()
