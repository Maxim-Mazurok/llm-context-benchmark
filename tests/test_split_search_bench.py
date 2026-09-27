import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('split_bench',Path(__file__).resolve().parents[1]/'scripts/split-search-bench.py')
BENCH=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
REPEAT_SPEC=importlib.util.spec_from_file_location('repeat_probe',Path(__file__).resolve().parents[1]/'scripts/repeat-stream-probe.py')
REPEAT=importlib.util.module_from_spec(REPEAT_SPEC)
REPEAT_SPEC.loader.exec_module(REPEAT)


class FakeServer:
    def __init__(self, corrupt=False):
        self.cached=[]
        self.corrupt=corrupt

    def __call__(self, endpoint, payload=None, stream=False):
        prompt=payload['prompt']
        cache=len(self.cached) if payload['cache_prompt'] and prompt[:len(self.cached)]==self.cached else 0
        evaluated=len(prompt)-cache
        self.cached=list(prompt)
        timings={'prompt_n':evaluated,'prompt_ms':max(1,evaluated)*10,'prompt_per_second':100,'cache_n':cache}
        if stream:
            events=[{'tokens':[99]} for _ in range(256)]
            events.append({'stop':True,'timings':timings})
            return io.BytesIO(b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events))
        token=payload['logit_bias'][0][0]
        return {'tokens':[token+1 if self.corrupt else token],'timings':timings}


class SplitBenchmarkTests(unittest.TestCase):
    def test_explicit_trial_endpoint(self):
        with patch.object(BENCH,'BASE_URL','http://127.0.0.1:8081/'),patch.object(BENCH,'urlopen',return_value=io.BytesIO(b'{"status":"ok"}')) as opened:
            self.assertEqual(BENCH.request('/health'),{'status':'ok'})
            self.assertEqual(opened.call_args.args[0].full_url,'http://127.0.0.1:8081/health')

    def test_fixed_path_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'trial'; run.mkdir()
            workload={'workload_sha256':'fixed','probes':[
                {'input_tokens':[1,2,3],'continuation':[11,12,13]},
                {'input_tokens':[4,5],'continuation':[14,15]}]}
            with patch.object(BENCH,'prepare_workload',return_value=workload),patch.object(BENCH,'request',FakeServer()):
                BENCH.main(run)
            result=json.loads((run/'summary.json').read_text())
            self.assertEqual(result['phases'],2)
            self.assertEqual(json.loads((run/'fixed-sequence.json').read_text())['tokens'],[1,2,3,11,12,13,4,5,14,15])
            rows=[json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['timings']['prompt_n']==1 for r in rows if r['kind']=='controlled_decode'))

    def test_rejects_wrong_forced_token(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            workload={'workload_sha256':'fixed','probes':[{'input_tokens':[1,2],'continuation':[11,12]}]}
            with patch.object(BENCH,'prepare_workload',return_value=workload),patch.object(BENCH,'request',FakeServer(corrupt=True)):
                with self.assertRaisesRegex(RuntimeError,'Unexpected forced completion'):
                    BENCH.main(run)

    def test_unscored_streaming_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source'; source.mkdir()
            run=Path(directory)/'repeat'; run.mkdir()
            BENCH.write_json(source/'fixed-sequence.json',{'tokens':[1,2,3]})
            BENCH.write_json(source/'streamed-decode.json',{'generated_tokens':[99]*256})
            BENCH.write_json(run/'launch.json',{'reused_from':str(source)})
            fake=FakeServer()
            def request(endpoint,payload=None,stream=False):
                return [{'is_processing':False}] if endpoint=='/slots' else fake(endpoint,payload,stream)
            with patch.object(REPEAT.bench,'request',request): REPEAT.main(run)
            result=json.loads((run/'streamed-repeat.json').read_text())
            self.assertEqual(result['matching_output_prefix_tokens'],256)
            self.assertEqual(result['timings']['prompt_n'],3)
            self.assertFalse((run/'summary.json').exists())


if __name__=='__main__': unittest.main()
