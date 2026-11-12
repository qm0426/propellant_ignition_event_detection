from __future__ import annotations


import hashlib

import importlib.metadata

import importlib.util


import functools

import json

import math

import os

from pathlib import Path



import sys



import numpy as np

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

class Blocked(RuntimeError):
    """A required real input, specification, dependency or checkpoint is missing."""

def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

def digest(obj):
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True, ensure_ascii=True).encode()).hexdigest()

@functools.lru_cache(maxsize=250000)
def _stat_cached_hash(path, size, mtime_ns, ctime_ns):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def file_hash(path):
    # Process-local only. Every lookup re-stats the input; modified files miss
    # the cache, and a mutation while hashing is rejected. No persistent trust
    # based on filenames, and no loss of content hashes in run provenance.
    path=Path(path).resolve();before=path.stat()
    result=_stat_cached_hash(str(path),before.st_size,before.st_mtime_ns,before.st_ctime_ns)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise RuntimeError(f'File modified while fingerprinting: {path}')
    return result

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, encoding='utf-8-sig')

def read_table(path):
    path = Path(path)
    if not path.is_file():
        raise Blocked(f'Missing table: {path}')
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)

def require_columns(df, cols, name):
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f'{name}: missing columns {missing}')

def boolean(value):
    if value is True or value == 1 or str(value).strip().lower() == 'true':
        return True
    if value is False or value == 0 or str(value).strip().lower() == 'false':
        return False
    raise ValueError(f'Expected explicit boolean, got {value!r}')

def load_legacy(filename, source_root=None):
    root = Path(source_root or PROJECT / 'method_core')
    path = (root / filename).resolve()
    if not path.is_file():
        raise Blocked(f'Missing legacy module: {path}')
    name = '_ignition_revision_legacy_' + file_hash(path)[:16] + '_' + path.stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]

def load_config(path):
    path = Path(path).resolve()
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.json':
        cfg = json.loads(text)
    else:
        import yaml
        cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a mapping')
    cfg['_config_path'] = str(path)
    base = Path(cfg.get('project_root') or PROJECT)
    cfg['project_root'] = str(base.resolve() if base.is_absolute() else (PROJECT / base).resolve())
    for key in ('legacy_root','upstream_root'):
        cfg[key] = str(resolve(cfg, cfg.get(key, 'method_core')))
    if cfg.get('detector', {}).get('model'):
        cfg['detector']['model'] = str(resolve(cfg, cfg['detector']['model']))
    return cfg

def resolve(cfg, value):
    if value is None:
        raise Blocked('Required path has not been configured')
    p = Path(value)
    return p.resolve() if p.is_absolute() else (Path(cfg['project_root']) / p).resolve()
