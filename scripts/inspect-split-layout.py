"""Read GGUF headers only and estimate weight placement, excluding runtime buffers."""
import argparse
import bisect
import collections
import json
import struct
from pathlib import Path


def tensors(path):
    with path.open('rb') as file:
        def number(fmt): return struct.unpack('<'+fmt,file.read(struct.calcsize('<'+fmt)))[0]
        def string(): return file.read(number('Q')).decode('utf-8')
        def value(kind):
            if kind==8: return string()
            if kind==9:
                subtype,count=number('I'),number('Q')
                for _ in range(count): value(subtype)
                return None
            return number({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[kind])
        if file.read(4)!=b'GGUF': raise ValueError('Not GGUF')
        version=number('I'); count=number('Q'); metadata_count=number('Q'); alignment=32
        for _ in range(metadata_count):
            key=string(); item=value(number('I'))
            if key=='general.alignment': alignment=item
        items=[]
        for _ in range(count):
            name=string(); dims=[number('Q') for _ in range(number('I'))]
            kind=number('I'); offset=number('Q'); items.append((offset,name))
        data=(file.tell()+alignment-1)//alignment*alignment
        items.sort()
        return [(name,(items[i+1][0] if i+1<len(items) else path.stat().st_size-data)-offset)
                for i,(offset,name) in enumerate(items)]


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('model_directory',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args()
    weights=[tensor for file in sorted(args.model_directory.glob('*.gguf')) for tensor in tensors(file)]
    results=[]
    for ngl,split in ((35,[5,26,4]),(27,[5,18,4]),(25,[5,16,4]),(23,[5,14,4]),(21,[5,12,4]),(19,[5,10,4]),(17,[5,8,4])):
        boundaries=[]; cumulative=0
        for part in split:
            cumulative+=part; boundaries.append(cumulative/sum(split))
        start=49-ngl
        def device(layer):
            return 'parent_cpu' if layer<start else ['worker_gpu','worker_cpu','parent_gpu'][bisect.bisect_right(boundaries,(layer-start)/ngl)]
        totals=collections.Counter()
        for name,size in weights:
            if name=='per_layer_token_embd.weight': target='lazy_lookup_table'
            elif name.startswith('blk.'): target=device(int(name.split('.')[1]))
            elif name=='token_embd.weight': target='parent_cpu'
            else: target=device(48)
            totals[target]+=size
        results.append({'gpu_layers':ngl,'split':split,'parent_cpu_layers':start,
                        'weight_gib':{k:round(v/2**30,4) for k,v in totals.items()}})
    args.output.write_text(json.dumps({'note':'Header-based estimate; runtime placement and cache/buffers must be measured',
                                      'candidates':results},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))
