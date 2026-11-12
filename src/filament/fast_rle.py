"""Canonical original-space COCO RLE from a bounded binary ROI without full canvas."""
import numpy as np
from pycocotools import mask as cm

def encode_binary_roi(binary,box,shape=(2048,2048)):
    a,b,c,d=map(int,box);height,width=shape
    if not (0<=a<=c<=width and 0<=b<=d<=height) or binary.shape!=(d-b,c-a):raise ValueError('ROI outside canvas')
    # Fortran-order intervals per column, then coalesce exact touching runs.
    change=np.diff(np.pad(np.asarray(binary,dtype=np.int8),((1,1),(0,0))),axis=0).T
    starts=np.argwhere(change==1);ends=np.argwhere(change==-1)
    assert len(starts)==len(ends)
    intervals=[]
    for start,end in zip(starts,ends):
        lo=(a+int(start[0]))*height+b+int(start[1]);hi=(a+int(end[0]))*height+b+int(end[1])
        if intervals and intervals[-1][1]==lo:intervals[-1][1]=hi
        else:intervals.append([lo,hi])
    counts=[];last=0
    for lo,hi in intervals:counts.extend([lo-last,hi-lo]);last=hi
    if last<height*width or not counts:counts.append(height*width-last)
    return cm.frPyObjects(dict(size=list(shape),counts=counts),height,width)

def encode_roi(prob,box,threshold=.5):return encode_binary_roi(prob>=threshold,box)

def encode_mask(mask):
    ys,xs=np.nonzero(mask)
    if not len(xs):return encode_binary_roi(np.zeros((0,0),bool),[0,0,0,0],mask.shape)
    a,b,c,d=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
    return encode_binary_roi(mask[b:d,a:c],[a,b,c,d],mask.shape)
