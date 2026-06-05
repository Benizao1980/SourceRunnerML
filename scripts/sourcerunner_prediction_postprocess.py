#!/usr/bin/env python3
"""
Post-process SourceRunner full-validation prediction outputs.

Adds compact prediction tables with isolate/sample metadata, merges optional legacy metadata
(e.g. year, ST, MLST clonal complex), writes summaries by metadata group, and exports plots.
Python 3.6 compatible.
"""
__version__ = "1.0.0"

import argparse, os, re, json
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True, help='human_predictions_bootstrap_ensemble.tsv')
    p.add_argument('--metadata', default=None, help='Optional original metadata CSV/TSV with id/isolate/year/ST/CC etc')
    p.add_argument('--model_comparison', default=None)
    p.add_argument('--bootstrap_metrics', default=None)
    p.add_argument('--outdir', required=True)
    p.add_argument('--id_col', default='id')
    p.add_argument('--top_n', type=int, default=30)
    p.add_argument('--lin_prefix_depths', default='6,7,8,10', help='Comma list of LINcode prefix depths to summarise')
    return p.parse_args()


def sniff_sep(path):
    if path.endswith('.tsv') or path.endswith('.txt'):
        return '\t'
    return ','


def clean_year(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if not s or s.lower() in ('nan','na','none','missing'):
        return np.nan
    m = re.search(r'(19|20)\d{2}', s)
    return m.group(0) if m else np.nan


def normalise_cols(df):
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('st (mlst)', 'st_mlst', 'mlst_st'):
            rename[c] = 'ST_MLST'
        elif cl in ('clonal_complex (mlst)', 'clonal complex (mlst)', 'cc_mlst', 'mlst_clonal_complex'):
            rename[c] = 'clonal_complex_MLST'
        elif cl == 'year':
            rename[c] = 'year'
        elif cl == 'country':
            rename[c] = 'country'
        elif cl == 'continent':
            rename[c] = 'continent'
        elif cl == 'region':
            rename[c] = 'region'
        elif cl in ('town_or_city', 'town/city'):
            rename[c] = 'town_or_city'
        elif cl == 'isolate':
            rename[c] = 'isolate'
        elif cl == 'id':
            rename[c] = 'id'
    return df.rename(columns=rename)


def compact_predictions(pred):
    camp_cols = [c for c in pred.columns if re.match(r'^CAMP\d+', str(c))]
    keep = [c for c in pred.columns if c not in camp_cols]
    out = pred[keep].copy()
    if 'clonal_complex' in out.columns and 'clonal_complex_cgMLST' not in out.columns:
        out = out.rename(columns={'clonal_complex':'clonal_complex_cgMLST'})
    if 'cgST' in out.columns:
        out = out.rename(columns={'cgST':'cgST_cgMLST'})
    return out, len(camp_cols)


def merge_metadata(compact, meta_path, id_col):
    if not meta_path:
        return compact, []
    meta = pd.read_csv(meta_path, sep=sniff_sep(meta_path), dtype=str, low_memory=False)
    meta = normalise_cols(meta)
    # de-duplicate metadata to avoid row inflation
    key = id_col if id_col in meta.columns and id_col in compact.columns else None
    if key is None and 'isolate' in meta.columns and 'isolate' in compact.columns:
        key = 'isolate'
    if key is None:
        return compact, []
    meta = meta.drop_duplicates(subset=[key])
    wanted = [key]
    for c in ['year','ST_MLST','clonal_complex_MLST','country','continent','region','town_or_city','isolation_date','source','disease','species']:
        if c in meta.columns and c not in wanted:
            wanted.append(c)
    meta = meta[wanted].copy()
    if 'year' in meta.columns:
        meta['year'] = meta['year'].map(clean_year)
    # avoid overwriting already-present columns except key
    rename = {}
    for c in meta.columns:
        if c != key and c in compact.columns:
            rename[c] = c + '_metadata'
    meta = meta.rename(columns=rename)
    merged = compact.merge(meta, on=key, how='left', validate='m:1')
    added = [c for c in merged.columns if c not in compact.columns]
    return merged, added


def lin_prefix(lin, depth):
    if pd.isna(lin):
        return np.nan
    s = str(lin)
    if s.lower() in ('nan','na','none','missing',''):
        return np.nan
    parts = s.split('_')
    if len(parts) < depth:
        return s
    return '_'.join(parts[:depth])


def summarise_by(df, group_col, pred_col, out_path):
    d = df.copy()
    if group_col not in d.columns or pred_col not in d.columns:
        return None
    d[group_col] = d[group_col].replace('', np.nan).fillna('Missing')
    x = (d.groupby([group_col, pred_col]).size().rename('n').reset_index())
    totals = d.groupby(group_col).size().rename('group_total').reset_index()
    x = x.merge(totals, on=group_col, how='left')
    x['percent_within_group'] = 100 * x['n'] / x['group_total']
    metrics = []
    for m in ['max_probability','max_probability_sd_across_bootstrap_models','bootstrap_consensus','normalised_entropy']:
        if m in d.columns:
            z = d.groupby(group_col)[m].agg(['mean','median']).reset_index()
            z.columns = [group_col, m+'_mean', m+'_median']
            metrics.append(z)
    for z in metrics:
        x = x.merge(z, on=group_col, how='left')
    x = x.sort_values(['group_total', group_col, 'n'], ascending=[False, True, False])
    x.to_csv(out_path, sep='\t', index=False)
    return x


def safe_plot_bar(summary, out_png, title, max_groups=30):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # create stacked table top groups only
        if summary is None or summary.empty:
            return
        group_col = summary.columns[0]
        pred_col = summary.columns[1]
        totals = summary[[group_col, 'group_total']].drop_duplicates().sort_values('group_total', ascending=False).head(max_groups)
        s = summary[summary[group_col].isin(totals[group_col])]
        pivot = s.pivot_table(index=group_col, columns=pred_col, values='n', aggfunc='sum', fill_value=0)
        pivot = pivot.loc[totals[group_col]]
        ax = pivot.plot(kind='bar', stacked=True, figsize=(max(8, len(pivot)*0.35), 5))
        ax.set_title(title)
        ax.set_xlabel(group_col)
        ax.set_ylabel('Number of isolates')
        plt.xticks(rotation=75, ha='right')
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        with open(str(out_png)+'.error.txt','w') as fh:
            fh.write(str(e))


def plot_simple_counts(df, pred_col, out_png, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        vc = df[pred_col].value_counts()
        ax = vc.plot(kind='bar', figsize=(7,5))
        ax.set_title(title)
        ax.set_xlabel(pred_col)
        ax.set_ylabel('Number of isolates')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        with open(str(out_png)+'.error.txt','w') as fh: fh.write(str(e))


def plot_hist(df, col, out_png, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if col not in df.columns: return
        ax = df[col].dropna().astype(float).plot(kind='hist', bins=40, figsize=(7,5))
        ax.set_title(title)
        ax.set_xlabel(col)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        with open(str(out_png)+'.error.txt','w') as fh: fh.write(str(e))


def plot_model_comparison(path, out_png):
    try:
        if not path or not os.path.exists(path): return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d = pd.read_csv(path, sep='\t')
        metric = 'balanced_accuracy' if 'balanced_accuracy' in d.columns else d.columns[1]
        xcol = 'model' if 'model' in d.columns else d.columns[0]
        d = d.sort_values(metric, ascending=False)
        ax = d.plot(x=xcol, y=metric, kind='bar', legend=False, figsize=(7,5))
        ax.set_ylim(0, 1)
        ax.set_title('Cross-validation model comparison')
        ax.set_ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        with open(str(out_png)+'.error.txt','w') as fh: fh.write(str(e))


def write_overall_summary(df, out_path, n_loci, added_cols):
    with open(out_path, 'w') as fh:
        fh.write('SourceRunner prediction post-processing summary\n\n')
        fh.write('Rows: %d\n' % len(df))
        fh.write('CAMP loci removed from compact table: %d\n' % n_loci)
        fh.write('Metadata columns added from external metadata: %s\n\n' % (', '.join(added_cols) if added_cols else 'none'))
        for col in ['predicted_filtered','predicted_source']:
            if col in df.columns:
                fh.write('%s counts:\n' % col)
                vc = df[col].value_counts(dropna=False)
                for k,v in vc.items():
                    fh.write('  %s\t%d\t%.2f%%\n' % (str(k), int(v), 100*v/len(df)))
                fh.write('\n')
        for col in ['max_probability','max_probability_sd_across_bootstrap_models','bootstrap_consensus','normalised_entropy']:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce')
                fh.write('%s: mean=%.4f median=%.4f min=%.4f max=%.4f\n' % (col, vals.mean(), vals.median(), vals.min(), vals.max()))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions, sep='\t', dtype=str, low_memory=False)
    compact, n_loci = compact_predictions(pred)
    compact, added = merge_metadata(compact, args.metadata, args.id_col)

    # Make sure numeric metrics are numeric
    for c in ['max_probability','max_probability_sd_across_bootstrap_models','bootstrap_consensus','normalised_entropy']:
        if c in compact.columns:
            compact[c] = pd.to_numeric(compact[c], errors='coerce')

    # LIN prefixes
    if 'LINcode' in compact.columns:
        for d in [int(x) for x in args.lin_prefix_depths.split(',') if x.strip()]:
            compact['LINcode_prefix_%d' % d] = compact['LINcode'].map(lambda x: lin_prefix(x, d))

    # Compact prediction table
    compact_path = outdir / 'human_predictions_enriched_metadata_compact.tsv'
    compact.to_csv(compact_path, sep='\t', index=False)

    # probability-only compact table with metadata
    prob_cols = [c for c in compact.columns if c.startswith('prob_mean_') or c.startswith('prob_sd_')]
    meta_cols = [c for c in ['id','isolate','country','year','ST_MLST','clonal_complex_MLST','cgST_cgMLST','clonal_complex_cgMLST','LINcode','predicted_source','predicted_filtered','max_probability','max_probability_sd_across_bootstrap_models','bootstrap_consensus','normalised_entropy'] if c in compact.columns]
    compact[meta_cols + prob_cols].to_csv(outdir / 'human_predictions_enriched_probabilities.tsv', sep='\t', index=False)

    # Summaries
    summary_cols = ['country','year','ST_MLST','clonal_complex_MLST','cgST_cgMLST','clonal_complex_cgMLST','LINcode']
    summary_cols += [c for c in compact.columns if c.startswith('LINcode_prefix_')]
    for group_col in summary_cols:
        if group_col in compact.columns:
            for pred_col, label in [('predicted_filtered','filtered'), ('predicted_source','raw')]:
                if pred_col in compact.columns:
                    s = summarise_by(compact, group_col, pred_col, outdir / ('summary_by_%s_%s.tsv' % (group_col, label)))
                    if label == 'filtered' and group_col in ['country','year','ST_MLST','clonal_complex_MLST','clonal_complex_cgMLST','LINcode_prefix_7','LINcode_prefix_8']:
                        safe_plot_bar(s, outdir / ('plot_summary_by_%s_%s.png' % (group_col, label)), 'Predicted source by %s' % group_col, args.top_n)

    # Overall plots
    if 'predicted_filtered' in compact.columns:
        plot_simple_counts(compact, 'predicted_filtered', outdir/'plot_overall_filtered_predictions.png', 'Overall filtered predictions')
    if 'predicted_source' in compact.columns:
        plot_simple_counts(compact, 'predicted_source', outdir/'plot_overall_raw_predictions.png', 'Overall raw predictions')
    for col in ['max_probability','max_probability_sd_across_bootstrap_models','bootstrap_consensus','normalised_entropy']:
        plot_hist(compact, col, outdir/('plot_%s_histogram.png' % col), col)
    plot_model_comparison(args.model_comparison, outdir/'plot_model_comparison_balanced_accuracy.png')

    write_overall_summary(compact, outdir/'postprocess_summary.txt', n_loci, added)
    with open(outdir/'postprocess_summary.json','w') as fh:
        json.dump({'rows': int(len(compact)), 'camp_loci_removed': int(n_loci), 'metadata_columns_added': added, 'output_compact': str(compact_path)}, fh, indent=2)
    print('Wrote:', compact_path)

if __name__ == '__main__':
    main()
