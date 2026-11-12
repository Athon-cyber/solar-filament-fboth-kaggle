"""Original exclusive pixel ownership used by OPT-031."""
import numpy as np

def exclusive_soft_groups(groups,threshold=.5):
    priority=np.zeros((2048,2048),np.float32);owner=np.full((2048,2048),-1,np.int16)
    for i,g in enumerate(groups):
        a,b,c,d=g['box'];p=g['prob'];sl=np.s_[b:d,a:c];quality=p*g['score'];eligible=(p>=threshold)&(quality>priority[sl]);priority[sl][eligible]=quality[eligible];owner[sl][eligible]=i
    output=[]
    for i,g in enumerate(groups):
        a,b,c,d=g['box'];mask=(owner[b:d,a:c]==i)
        if not mask.any():continue
        from .fast_rle import encode_binary_roi
        encoded=encode_binary_roi(mask,g['box']);output.append(dict(confidence=g['score'],rle=dict(size=[2048,2048],counts=encoded['counts'].decode('ascii'))))
    return output
