"""One ROI reader for stored crops and virtual crops in original frame coordinates."""
from PIL import Image
import numpy as np
from functools import lru_cache
from .common import resolve


@lru_cache(maxsize=256)
def _full_frame(path,size,mtime_ns):
    with Image.open(path) as image:return np.asarray(image.convert('RGB')).copy()

def read_roi(row, cfg):
    path=resolve(cfg,row.roi_path)
    with Image.open(path) as image:
        keys = ['crop_left', 'crop_top', 'crop_right_exclusive', 'crop_bottom_exclusive']
        supplied = [key in row and row.get(key) is not None and not np.isnan(float(row[key])) for key in keys]
        if any(supplied) and not all(supplied):
            raise ValueError('Virtual ROI requires all four crop boundaries')
        if all(supplied):
            box = tuple(int(row[key]) for key in keys)
            if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                raise ValueError(f'Invalid virtual ROI: {box}')
            stat=path.stat()
            return _full_frame(str(path),stat.st_size,stat.st_mtime_ns)[box[1]:box[3],box[0]:box[2]].copy()
        return np.asarray(image.convert('RGB')).copy()
