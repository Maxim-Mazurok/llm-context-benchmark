"""Unscored same-server streaming repeat to examine warm-state variation."""
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

spec=importlib.util.spec_from_file_location('split_bench',Path(__file__).with_name('split-search-bench.py'))
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def main(run):
    if (run/'streamed-repeat.json').exists():
        raise RuntimeError('Refusing to overwrite a prior measurement')
    launch=json.loads((run/'launch.json').read_text(encoding='utf-8-sig'))
    source=Path(launch['reused_from'])
    prefix=json.loads((source/'fixed-sequence.json').read_text())['tokens']
    slots=bench.request('/slots')
    if any(slot['is_processing'] for slot in slots):
        raise RuntimeError('Server is busy')
    bench.write_json(run/'run-state.json',{'status':'running','started_utc':datetime.now(timezone.utc).isoformat()})
    bench.write_json(run/'benchmark-config.json',{'protocol':'unscored streaming repeat',
        'source_run':str(source),'prompt_sha256':bench.digest(prefix),'prompt_tokens':len(prefix),
        'cache_prompt':False,'generated_tokens':256,'temperature':0,'seed':1234})
    payload={'prompt':prefix,'n_predict':256,'cache_prompt':False,'id_slot':0,
             'temperature':0,'seed':1234,'ignore_eos':True,'return_tokens':True,
             'stream':True,'logit_bias':[]}
    timestamps=[]; generated=[]; final=None
    started=time.perf_counter()
    with bench.request('/completion',payload,stream=True) as response:
        for line in response:
            if (run/'abort.json').exists():
                raise RuntimeError('Telemetry requested abort')
            if not line.startswith(b'data: '): continue
            data=line[6:].strip()
            if data==b'[DONE]': continue
            event=json.loads(data)
            if event.get('error') or event.get('truncated'): raise RuntimeError(event)
            for token in event.get('tokens',[]):
                generated.append(token); timestamps.append(time.perf_counter()-started)
            if event.get('stop'): final=event
    if len(generated)!=256 or final is None or final['timings']['prompt_n']!=len(prefix):
        raise RuntimeError('Unexpected token accounting in streaming repeat')
    prior=json.loads((source/'streamed-decode.json').read_text())
    matching=0
    while matching<256 and generated[matching]==prior['generated_tokens'][matching]: matching+=1
    result={'source_run':str(source),'prompt_sha256':bench.digest(prefix),
            'generated_tokens':generated,'generated_sha256':bench.digest(generated),
            'timestamps_s':timestamps,'timings':final['timings'],'wall_s':time.perf_counter()-started,
            'stream_decode_tps':255/(timestamps[-1]-timestamps[0]),
            'matching_output_prefix_tokens':matching,
            'note':'Unscored same-server repeat; fresh prompt evaluation but already exercised weights and lookup data.'}
    bench.write_json(run/'streamed-repeat.json',result)
    bench.write_json(run/'run-state.json',{'status':'completed','completed_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({k:v for k,v in result.items() if k not in ('generated_tokens','timestamps_s')},indent=2),flush=True)


if __name__=='__main__':
    directory=Path(sys.argv[1])
    try:
        main(directory)
    except Exception as error:
        bench.write_json(directory/'run-state.json',{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
        raise
