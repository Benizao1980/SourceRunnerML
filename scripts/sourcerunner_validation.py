"""Explicit group-contained CV contracts; no inferred biological grouping."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class ValidationDesignError(ValueError):
    """A validation contract failed; never silently downgrade to isolate CV."""


def load_validation_config(path):
    if not path:
        return None
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    allowed = {'strategy', 'group_column', 'id_column', 'collision_policy'}
    if not isinstance(config, dict) or set(config)-allowed:
        raise ValidationDesignError('Validation configuration has unknown keys or is not an object')
    if config.get('strategy') != 'grouped':
        raise ValidationDesignError('This increment supports strategy=grouped; omit config for default stratified CV')
    for key in ('group_column', 'id_column'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValidationDesignError(f'Grouped validation requires {key}')
    if config['group_column'] == config['id_column']:
        raise ValidationDesignError('Group and sample-ID columns must be distinct')
    config.setdefault('collision_policy', 'error')
    if config['collision_policy'] not in ('error', 'warn'):
        raise ValidationDesignError('collision_policy must be error or warn')
    return config


def metadata_values(frame, column, unique=False):
    if column not in frame:
        raise ValidationDesignError(f'Metadata lacks column {column}')
    raw = frame[column]
    if raw.isna().any() or raw.astype(str).str.strip().eq('').any():
        raise ValidationDesignError(f'Missing/blank values in {column}; resolve or explicitly exclude upstream')
    values = raw.astype(str).str.strip().to_numpy()
    if unique and len(set(values)) != len(values):
        raise ValidationDesignError(f'Duplicate sample identifiers in {column}')
    return values


def grouped_splits(X, y, groups, folds, seed):
    """No group crosses any boundary; require all sources in both partitions.

    StratifiedGroupKFold is a heuristic. Failure is not proof that every possible
    assignment is impossible; users should redesign/reduce folds explicitly.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    if len(groups) != len(y) or len(X) != len(y):
        raise ValidationDesignError('Rows, source labels and groups differ in length')
    if folds < 2 or len(np.unique(groups)) < folds:
        raise ValidationDesignError('Need at least two folds and at least one distinct group per fold')
    classes = set(np.unique(y))
    if len(classes) < 2:
        raise ValidationDesignError('Validation requires at least two source classes')
    for cls in classes:
        if len(np.unique(groups[y == cls])) < folds:
            raise ValidationDesignError(f'Source class {cls} spans fewer than {folds} groups; reduce folds or obtain independent groups')
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups))
    for fold, (tr, te) in enumerate(splits):
        if set(groups[tr]) & set(groups[te]):
            raise ValidationDesignError(f'Fold {fold} splits a declared group')
        if set(y[tr]) != classes or set(y[te]) != classes:
            raise ValidationDesignError(f'Fold {fold} lacks a required source in train/test; splitter heuristic failed, redesign folds')
    return splits


def row_fingerprints(values):
    # Current numeric path only. Canonicalize missingness and signed zero; no
    # source/group/ID column participates. No all-pairs matrix is allocated.
    matrix = np.asarray(values, dtype='<f8').copy()
    matrix[matrix == 0] = 0.
    matrix[np.isnan(matrix)] = np.nan
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in matrix]


def collision_audit(train, test, policy='error', stage='raw'):
    a, b = row_fingerprints(train), row_fingerprints(test)
    overlap = set(a) & set(b)
    record = {'stage': stage, 'distinct_shared_profiles': len(overlap),
              'test_rows_matching_training': sum(x in overlap for x in b)}
    if overlap and policy == 'error':
        raise ValidationDesignError(f'{stage}: train/test profile collision ({len(overlap)} distinct profiles); review groups or explicitly choose collision_policy=warn')
    if overlap and policy == "warn":
        print(f"WARNING: {stage}: {len(overlap)} shared train/test profiles; explicitly allowed by collision_policy=warn", file=sys.stderr)
    return record


def audit_training_transform(X, tr, te, missingness, policy):
    from sourcerunner_features import AllelePreprocessor
    pre = AllelePreprocessor(missingness).fit(X.iloc[tr])
    return collision_audit(pre.transform(X.iloc[tr]), pre.transform(X.iloc[te]),
                           policy, 'training-fitted transformed features')


def prepare_grouped_validation(train, X, y, config, folds, seed, source_column=None):
    ids = metadata_values(train, config['id_column'], unique=True)
    groups = metadata_values(train, config['group_column'])
    metadata_columns = {config['group_column'], config['id_column']}
    if source_column is not None:
        metadata_columns.add(source_column)
    if metadata_columns & set(X.columns):
        raise ValidationDesignError('Validation metadata columns must not be selected genomic loci')
    splits = grouped_splits(X, y, groups, folds, seed)
    records, audits = [], []
    for fold, (tr, te) in enumerate(splits):
        audits.append(collision_audit(X.iloc[tr], X.iloc[te], config['collision_policy'], 'raw numeric profiles'))
        for role, indices in (('train', tr), ('test', te)):
            records.extend({'fold': fold, 'role': role, 'sample_id': ids[i],
                            'group': groups[i], 'encoded_source': int(y[i])} for i in indices)
    return ids, groups, splits, pd.DataFrame(records), audits
