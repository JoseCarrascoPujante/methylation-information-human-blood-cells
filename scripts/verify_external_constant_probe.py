#!/usr/bin/env python3
"""Read one probe from a complete public intensity gzip and validate it to EOF."""
import argparse
import csv
import gzip
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--signals',type=Path,required=True)
    p.add_argument('--probe',default='cg24418853')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.signals.resolve()
    matches=[];rows=0;decompressed=0
    needle=(args.probe+'\t').encode()
    with gzip.open(source,'rb') as f:
        header_bytes=f.readline();decompressed+=len(header_bytes)
        for line in f:
            rows+=1;decompressed+=len(line)
            if line.startswith(needle):matches.append(line)
    if len(matches)!=1:raise ValueError(f'Probe appeared {len(matches)} times.')
    h=header_bytes.decode().rstrip('\r\n').split('\t')
    values=matches[0].decode().rstrip('\r\n').split('\t')
    if len(h)!=len(values) or (len(h)-1)%3:raise ValueError('Signal schema mismatch.')
    data=[]
    for col in range(1,len(h),3):
        array=h[col].removesuffix(' Unmethylated signal')
        assert h[col+1]==array+' Methylated signal'
        assert h[col+2]==array+' Detection Pval'
        u,m,det=map(float,values[col:col+3])
        data.append({'probe_id':args.probe,'array_id':array,'unmethylated_signal':u,
                     'methylated_signal':m,'detection_p':det,'beta':m/(m+u+100)})
    with (out/'probe_signals_and_beta.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'original_header_and_probe.tsv').write_bytes(header_bytes+matches[0])
    unique=sorted(set(r['beta'] for r in data))
    print(json.dumps({'probe':args.probe,'array_columns':len(data),'unique_beta_values':unique,
                      'constant_across_all_arrays':len(unique)==1}),flush=True)


if __name__=='__main__':main()
