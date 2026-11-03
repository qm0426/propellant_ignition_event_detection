"""NumPy equivalent of cython_bbox.bbox_overlaps (inclusive +1 convention).

ByteTrack's author matching code uses this pixel-inclusive IoU, which differs
from this project's continuous-coordinate IoU. Keep the author convention.
"""
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    a=np.asarray(boxes,dtype=float).reshape(-1,4)
    b=np.asarray(query_boxes,dtype=float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2])
    hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo+1.,0.).prod(axis=2)
    area_a=(a[:,2:]-a[:,:2]+1.).prod(axis=1)
    area_b=(b[:,2:]-b[:,:2]+1.).prod(axis=1)
    return inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,np.finfo(float).eps)
