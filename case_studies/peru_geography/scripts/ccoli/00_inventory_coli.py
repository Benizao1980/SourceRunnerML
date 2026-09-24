#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from coli_common import read_table,filter_species,normalise_source,detect_loci

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--global-source',required=True); p.add_argument('--peru-source',required=True); p.add_argument('--human',required=True)
    p.add_argument('--outdir',required=True); p.add_argument('--id-col',default='id'); p.add_argument('--source-col',default='source')
    p.add_argument('--country-col',default='country'); p.add_argument('--species-col',default='species'); p.add_argument('--species-value',default='C. coli')
    p.add_argument('--loci-prefix',default='CAMP'); p.add_argument('--loci-regex',default=''); p.add_argument('--contig-dirs',default='')
    a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    tabs={k:filter_species(read_table(v),a.species_col,a.species_value) for k,v in [('global',a.global_source),('peru_source',a.peru_source),('human',a.human)]}
    for name,d in tabs.items():
        if a.id_col not in d: raise ValueError(f'{name}: missing ID column {a.id_col!r}')
        d[a.id_col]=d[a.id_col].astype(str)
        if a.source_col in d: d['__source_norm__']=d[a.source_col].map(normalise_source)
    loci=detect_loci(list(tabs.values()),a.loci_prefix,a.loci_regex)
    lines=[f'{n}\trows={len(d)}\tcolumns={len(d.columns)}' for n,d in tabs.items()]+[f'shared_loci\t{len(loci)}',f'first_locus\t{loci[0]}',f'last_locus\t{loci[-1]}']
    (out/'input_summary.txt').write_text('\n'.join(lines)+'\n')
    counts=[]
    for name,d in tabs.items():
        if '__source_norm__' in d:
            for src,n in d['__source_norm__'].value_counts(dropna=False).items(): counts.append([name,src,n])
    pd.DataFrame(counts,columns=['dataset','source','n']).to_csv(out/'source_counts.tsv',sep='\t',index=False)
    gid=set(tabs['global'][a.id_col]); pid=set(tabs['peru_source'][a.id_col]); hid=set(tabs['human'][a.id_col])
    pd.DataFrame([['global_vs_peru_source',len(gid&pid)],['global_vs_human',len(gid&hid)],['peru_source_vs_human',len(pid&hid)]],columns=['comparison','n_exact_id_overlap']).to_csv(out/'overlap_summary.tsv',sep='\t',index=False)
    fasta_map={}; dup=set()
    for ds in [x for x in a.contig_dirs.split(':') if x]:
        for ext in ('*.fasta','*.fa','*.fna'):
            for f in Path(ds).glob(ext):
                sid=f.stem.split('_',1)[0]
                if sid in fasta_map: dup.add(sid)
                else: fasta_map[sid]=str(f)
    rows=[]
    for name,d in tabs.items():
        for sid in d[a.id_col].astype(str): rows.append([name,sid,int(sid in fasta_map),fasta_map.get(sid,'')])
    pd.DataFrame(rows,columns=['dataset','id','has_contig','contig_path']).to_csv(out/'contig_coverage.tsv',sep='\t',index=False)
    (out/'duplicate_contig_ids.txt').write_text('\n'.join(sorted(dup))+('\n' if dup else ''))
    print((out/'input_summary.txt').read_text(),end=''); print('\nSource counts:'); print(pd.read_csv(out/'source_counts.tsv',sep='\t').to_string(index=False)); print('\nExact ID overlaps:'); print(pd.read_csv(out/'overlap_summary.tsv',sep='\t').to_string(index=False)); print(f'\nInventory written to {out}')
if __name__=='__main__': main()
