"""Fixed-token coding trace for placement screening; standard library only.

Teacher-forced decode evaluates an identical token path through cached
single-token completion requests. Its HTTP-inclusive rate is NOT conventional
streamed decode throughput, which is measured separately at the end.
"""
import argparse
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'http://127.0.0.1:8080'


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def request(endpoint, payload=None, stream=False):
    req = Request(BASE_URL.rstrip('/') + endpoint,
                  data=json.dumps(payload).encode() if payload is not None else None,
                  headers={'Content-Type': 'application/json'},
                  method='POST' if payload is not None else 'GET')
    response = urlopen(req, timeout=1200)
    if stream:
        return response
    with response:
        return json.load(response)


def prepare_workload(path):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    sources = ['src/llm_context_benchmark/adapters/llama_server.py',
               'src/llm_context_benchmark/runner.py',
               'src/llm_context_benchmark/reporting.py']
    documents = [{'path': name, 'text': (ROOT/name).read_text(encoding='utf-8')} for name in sources]
    text = ('You are reviewing a Python inference benchmark. Inspect the following source, '
            'identify correctness risks, propose a minimal patch, and explain tests.\n\n')
    text += '\n\n'.join(f'File: {d["path"]}\n```python\n{d["text"]}\n```' for d in documents)
    pool = request('/tokenize', {'content': text, 'add_special': True})['tokens']
    continuation = request('/tokenize', {'content': (
        '\nReview: preserve exact token accounting, validate the cached prefix, and keep '
        'prefill and decode timing separate. Add tests for early termination, empty '
        'responses, malformed events, and context boundaries.\n'
        'def validate_result(result):\n'
        '    assert result.generated_tokens is not None\n'
        '    assert result.prefill_finished >= result.prefill_started\n'
        '    return len(result.generated_tokens)\n'), 'add_special': False})['tokens']
    pool = (pool * (3584//len(pool)+1))[:3584]
    continuation = (continuation * (64//len(continuation)+1))[:64]
    offset = 0
    probes = []
    for count in (2048, 512, 1024):
        probes.append({'input_tokens': pool[offset:offset+count], 'continuation': continuation})
        offset += count
    workload = {'version': 2, 'probes': probes, 'source_documents': documents,
                'source_sha256': digest(documents), 'workload_sha256': digest(probes),
                'description': 'Fixed coding inputs with 64 forced continuation tokens per stage'}
    write_json(path, workload)
    return workload


def main(run):
    state_path = run/'run-state.json'
    write_json(state_path, {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()})
    workload = prepare_workload(run.parent/'workload-v2.json')
    write_json(run/'benchmark-config.json', {'workload_sha256': workload['workload_sha256'],
        'base_url': BASE_URL,
        'protocol_version': 3, 'warmup': '64-token protocol check, then full fixed trace prefill',
        'continuation_mode': 'logit_bias forced token; one-token cached requests',
        'seed': 1234, 'temperature': 0, 'free_decode_tokens': 256})
    records = []
    phases = []
    prefix = []

    def complete(prompt, expected, cache, phase, kind, expected_prompt_n=None):
        if (run/'abort.json').exists():
            raise RuntimeError('Telemetry requested abort: '+(run/'abort.json').read_text())
        payload = {'prompt': prompt, 'n_predict': 1, 'cache_prompt': cache,
                   'id_slot': 0, 'temperature': 0, 'seed': 1234, 'ignore_eos': True,
                   'return_tokens': True, 'logit_bias': [[expected, 1000000]]}
        start = time.perf_counter()
        result = request('/completion', payload)
        elapsed = time.perf_counter()-start
        tokens = result.get('tokens')
        timings = result.get('timings', {})
        if tokens != [expected] or result.get('truncated'):
            raise RuntimeError(f'Unexpected forced completion: {tokens}, expected {expected}')
        if expected_prompt_n is not None and timings.get('prompt_n') != expected_prompt_n:
            raise RuntimeError(f'Cache mismatch: expected {expected_prompt_n} evaluated tokens, got {timings}')
        record = {'phase': phase, 'kind': kind, 'context_tokens': len(prompt),
                  'prompt_sha256': digest(prompt), 'generated_tokens': tokens,
                  'wall_s': elapsed, 'timings': timings}
        records.append(record)
        with (run/'requests.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, separators=(',', ':'))+'\n')
        return record

    # Uniform warmup and protocol/cache validation, excluded from the score.
    warm = workload['probes'][0]['input_tokens'][:64]
    forced = workload['probes'][0]['continuation'][:2]
    complete(warm, forced[0], False, 0, 'warmup', len(warm))
    complete(warm+[forced[0]], forced[1], True, 0, 'warmup', 1)
    warm_trace=[]
    for probe in workload['probes']:
        warm_trace+=probe['input_tokens']+probe['continuation']
    warm_record=complete(warm_trace, forced[0], False, 0, 'full_trace_warmup', len(warm_trace))
    print(f"Full trace warmup: {len(warm_trace)} tokens in {warm_record['wall_s']:.1f}s (excluded)",flush=True)
    trace_start = time.perf_counter()
    for index, probe in enumerate(workload['probes'], 1):
        old_len = len(prefix)
        prefix += probe['input_tokens']
        continuation = probe['continuation']
        begin = time.perf_counter()
        prefill = complete(prefix, continuation[0], index>1, index, 'prefill',
                           len(probe['input_tokens'])+(1 if old_len else 0))
        prefix.append(continuation[0])
        steps = []
        for token in continuation[1:]:
            step = complete(prefix, token, True, index, 'controlled_decode', 1)
            steps.append(step)
            prefix.append(token)
        phase = {'phase': index, 'context_tokens': len(prefix),
                 'prefill_tokens': prefill['timings']['prompt_n'],
                 'prefill_s': prefill['timings']['prompt_ms']/1000,
                 'prefill_tps': prefill['timings']['prompt_per_second'],
                 'controlled_decode_tokens': len(steps),
                 'controlled_decode_s': sum(s['wall_s'] for s in steps),
                 'controlled_decode_tps': len(steps)/sum(s['wall_s'] for s in steps),
                 'phase_wall_s': time.perf_counter()-begin,
                 'sequence_sha256': digest(prefix)}
        phases.append(phase)
        write_json(run/'phases.json', phases)
        print(f"Stage {index}: ctx={len(prefix)} prefill={phase['prefill_tps']:.2f} "
              f"controlled_decode={phase['controlled_decode_tps']:.2f} tok/s "
              f"wall={phase['phase_wall_s']:.1f}s", flush=True)
    trace_wall = time.perf_counter()-trace_start
    write_json(run/'fixed-sequence.json', {'tokens': prefix, 'sha256': digest(prefix)})

    # Conventional streaming probe: same prompt for all placements, unconstrained output.
    payload = {'prompt': prefix, 'n_predict': 256, 'cache_prompt': True, 'id_slot': 0,
               'temperature': 0, 'seed': 1234, 'ignore_eos': True, 'return_tokens': True,
               'stream': True, 'logit_bias': []}
    timestamps = []
    generated = []
    final = None
    started = time.perf_counter()
    with request('/completion', payload, stream=True) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                continue
            event = json.loads(data)
            if event.get('error') or event.get('truncated'):
                raise RuntimeError(event)
            for token in event.get('tokens', []):
                timestamps.append(time.perf_counter()-started)
                generated.append(token)
            if event.get('stop'):
                final = event
    if len(generated) != 256 or final is None or final['timings']['prompt_n'] != 1:
        raise RuntimeError('Invalid normal streaming probe or unexpected cache reprocessing')
    stream_result = {'prompt_sha256': digest(prefix), 'generated_tokens': generated,
                     'generated_sha256': digest(generated), 'timestamps_s': timestamps,
                     'timings': final['timings'], 'wall_s': time.perf_counter()-started,
                     'stream_decode_tps': 255/(timestamps[-1]-timestamps[0])}
    write_json(run/'streamed-decode.json', stream_result)
    summary = {'workload_sha256': workload['workload_sha256'],
               'protocol_version': 3, 'full_trace_warmup_s': warm_record['wall_s'],
               'fixed_sequence_sha256': digest(prefix), 'trace_wall_s': trace_wall,
               'prefill_tps': sum(p['prefill_tokens'] for p in phases)/sum(p['prefill_s'] for p in phases),
               'controlled_decode_tps': sum(p['controlled_decode_tokens'] for p in phases)/sum(p['controlled_decode_s'] for p in phases),
               'stream_decode_tps': stream_result['stream_decode_tps'],
               'max_context': len(prefix)+len(generated), 'phases': len(phases)}
    write_json(run/'summary.json', summary)
    write_json(state_path, {'status': 'completed', 'completed_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--base-url', default=BASE_URL,
                        help='Explicit trial API endpoint; do not aim benchmarks at an active production server')
    args = parser.parse_args()
    BASE_URL = args.base_url.rstrip('/')
    try:
        main(args.run_directory)
    except Exception as error:
        write_json(args.run_directory/'run-state.json', {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
        raise
