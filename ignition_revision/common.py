from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import functools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
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
    return pd.read_csv(path)


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
    root = Path(source_root or (PROJECT if filename in ('test_locate_anything.py','postprocess_detection_txt_modified.py') else PROJECT / 'latest_experiment_scripts_20261018'))
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
            # Legacy entry points import their framework by its old top-level name.
            # Bind their functions to the same frozen source root instead of a future edited root copy.
            if filename != 'run_ignition_ablation_experiments_revised.py' and hasattr(module,'framework'):
                module.framework = load_legacy('run_ignition_ablation_experiments_revised.py', root)
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
    cfg.setdefault('project_root', str(PROJECT))
    return cfg


def resolve(cfg, value):
    if value is None:
        raise Blocked('Required path has not been configured')
    p = Path(value)
    return p.resolve() if p.is_absolute() else (Path(cfg['project_root']) / p).resolve()


def input_fingerprints(cfg):
    """Hash explicit table, frame, checkpoint and source inputs, including referenced ROI images."""
    # The adapter just read these files; retain that content snapshot without a second full ROI scan.
    result = dict(cfg.get('legacy_source_hashes',{})) if cfg.get('legacy_input_audit') else {}
    def visit(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                if k not in ('output_dir', '_config_path', 'legacy_source_hashes', 'legacy_input_provenance'):
                    visit(v, key + '.' + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                visit(v, f'{key}.{i}')
        elif isinstance(value, str):
            if value in result:return
            try:
                p = resolve(cfg, value)
                if p.is_file(): result[str(p)] = file_hash(p)
                elif p.is_dir() and key.endswith('source_root'):
                    for source in p.rglob('*.py'): result[str(source)] = file_hash(source)
            except (OSError, ValueError):
                pass
    visit(cfg)
    image_jobs={}
    for key in ('frames', 'rois', 'state_labels'):
        if cfg.get(key) and resolve(cfg, cfg[key]).is_file():
            table = read_table(resolve(cfg, cfg[key]))
            for column in ('image_path', 'roi_path', 'mask_path'):
                if column in table:
                    for value in table[column].dropna().unique():
                        if str(value) in result:continue
                        p = Path(value)
                        if not p.is_absolute():p=(resolve(cfg,cfg['dataset_dir']) if key=='frames' and cfg.get('dataset_dir') else Path(cfg['project_root']))/p
                        image_jobs[str(p)]=p
    if image_jobs:
        from concurrent.futures import ThreadPoolExecutor
        def fingerprint_image(p):
            p=p.resolve()
            if not p.is_file():raise Blocked(f'Missing image: {p}')
            return str(p),file_hash(p)
        if len(image_jobs)>1000:print(f'INPUT_CONTENT_CHECK {len(image_jobs)} image files, 8 I/O workers',flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for index,(path,h) in enumerate(pool.map(fingerprint_image,image_jobs.values()),1):
                result[path]=h
                if len(image_jobs)>1000 and index%20000==0:print(f'INPUT_CONTENT_CHECK {index}/{len(image_jobs)}',flush=True)
    if cfg.get('dataset_dir'):
        dataset_root=resolve(cfg,cfg['dataset_dir'])
        for name in ('frames.csv','boxes.csv','ignore_boxes.csv','dataset_meta.json','shared_tracks.csv','cohort.csv','inference_frames.csv'):
            p=dataset_root/name
            if p.is_file():result[str(p)]=file_hash(p)
        frame_manifest=dataset_root/'frames.csv'
        if frame_manifest.is_file():
            for value in read_table(frame_manifest).image_path.dropna().unique():
                p=Path(value);p=p if p.is_absolute() else dataset_root/p
                result[str(p.resolve())]=file_hash(p)
    root = Path(cfg.get('legacy_root', PROJECT / 'latest_experiment_scripts_20261018'))
    if not root.is_absolute():
        root = resolve(cfg, str(root))
    for name in sorted(p.name for p in root.glob('*.py')):
        if (root / name).is_file():
            result[str(root / name)] = file_hash(root / name)
    upstream = Path(cfg.get('upstream_root', PROJECT))
    if not upstream.is_absolute(): upstream = resolve(cfg, str(upstream))
    for name in ('test_locate_anything.py','postprocess_detection_txt_modified.py'):
        if (upstream/name).is_file(): result[str(upstream/name)] = file_hash(upstream/name)
    return result


class Run:
    def __init__(self, cfg, family, resume=False):
        self.cfg, self.family = cfg, family
        if cfg.get('_review_v2_scope'):
            from .review_v2 import validate_scope
            validate_scope(cfg)
        if not cfg.get('synthetic',False) and cfg.get('protocol_status')!='frozen' and not (cfg.get('evaluation_mode')=='legacy_reproduction' and cfg.get('legacy_input_audit')):
            raise Blocked('Real formal runs require protocol_status=frozen after data, development parameters and evaluation scope are confirmed; dry-run remains available')
        self.inputs = input_fingerprints(cfg)
        code = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
        # Reference changes affect evaluation identity, not the prediction identity.
        prediction_cfg = {k: v for k, v in cfg.items() if k not in ('labels', 'reference_events', 'reference_boxes', 'evaluation', '_config_path', 'output_dir')}
        pred_inputs = {k: v for k, v in self.inputs.items() if k not in {
            str(resolve(cfg, cfg[x])) for x in ('labels', 'reference_events', 'reference_boxes') if cfg.get(x)}}
        environment={'python':sys.version}
        for dependency in ('numpy','pandas','scipy','scikit-learn','opencv-python','torch','torchvision','ruptures','filterpy','lap'):
            try:environment[dependency]=importlib.metadata.version(dependency)
            except importlib.metadata.PackageNotFoundError:environment[dependency]=None
        self.prediction_id = digest([family, prediction_cfg, pred_inputs, code, environment])[:20]
        self.run_id = digest([family, cfg, self.inputs, code, environment])[:20]
        self.out = resolve(cfg, cfg.get('output_dir', 'revision_outputs')) / family / self.run_id
        self.skip = False
        if self.out.exists():
            meta_path = self.out / 'run_meta.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            if resume and meta.get('status') == 'completed':
                expected = meta.get('outputs', {})
                if expected and all((self.out / k).is_file() and file_hash(self.out / k) == v for k, v in expected.items()):
                    self.skip = True
                    return
            raise FileExistsError(f'Run already exists/incomplete; use a new output_dir (no overwrite): {self.out}')
        self.out.mkdir(parents=True)
        snapshot=self.out/'code_snapshot'
        for source in Path(__file__).parent.glob('*.py'):
            dest=snapshot/'ignition_revision'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        for source_path,source_hash in self.inputs.items():
            if Path(source_path).suffix=='.py':
                dest=snapshot/'external_sources'/(Path(source_path).stem+'_'+source_hash[:12]+'.py');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_path,dest)
        self.timings, self.statuses = [], []
        self.started = time.perf_counter()
        versions = {}
        for package in ('numpy','pandas','scipy','scikit-learn','opencv-python','torch','torchvision','ruptures','filterpy','lap'):
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package] = None
        self.meta = {'status': 'running', 'run_id': self.run_id, 'prediction_run_id': self.prediction_id,
                     'family': family, 'created_at': datetime.now(timezone.utc).isoformat(),
                     'synthetic': bool(cfg.get('synthetic', False)), 'input_hashes': self.inputs,
                     'code_hashes': code, 'versions': versions, 'hardware': {'platform': platform.platform(),
                     'processor': platform.processor(), 'cpu_count': os.cpu_count(), 'python': sys.version}}
        write_json(self.out / 'resolved_config.json', cfg)
        if str(cfg.get('device','cpu')).startswith('cuda'):
            import torch
            self.meta['hardware']['cuda_version']=torch.version.cuda
            self.meta['hardware']['gpu_name']=torch.cuda.get_device_name(cfg['device'])
            self.meta['hardware']['gpu_memory_bytes']=torch.cuda.get_device_properties(cfg['device']).total_memory
        self.save()

    def save(self):
        write_json(self.out / 'run_meta.json', self.meta)

    @contextlib.contextmanager
    def time(self, stage, units=1, unit='tracks', device='cpu', cache_mode='cold'):
        def sync():
            if str(device).startswith('cuda'):
                import torch
                torch.cuda.synchronize(device)
        sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            sync()
            seconds = time.perf_counter() - start
            self.timings.append({'stage': stage, 'device': device, 'N_units': units, 'unit': unit,
                                 'total_s': seconds, 'ms_per_unit': seconds * 1000 / units if units else None,
                                 'cache_mode': cache_mode})

    def record(self, method, status, reason=None):
        self.statuses.append({'method_id': method, 'status': status, 'reason': reason})
        write_csv(self.out / 'experiment_status.csv', self.statuses)

    def finish(self, error=None):
        write_csv(self.out / 'runtime_summary.csv', self.timings,
                  ['stage','device','N_units','unit','total_s','ms_per_unit','cache_mode'])
        self.meta['elapsed_s'] = time.perf_counter() - self.started
        self.meta['error'] = str(error) if error else None
        self.meta['status'] = ('failed' if error else 'partial' if any(s['status'] != 'completed' for s in self.statuses) else 'completed')
        self.meta['outputs'] = {str(p.relative_to(self.out)): file_hash(p) for p in self.out.rglob('*')
                                if p.is_file() and p.name != 'run_meta.json'}
        self.save()
