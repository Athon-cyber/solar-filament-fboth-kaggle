"""OPT009: TTA-per-fold then cross-fold instance association/voting, NMS, pixel ownership."""
import numpy as np
from pycocotools import mask as cm
from .fast_rle import encode_roi

def associate(candidates,iou_threshold=.5,ambiguity_margin=.1,threshold=.5):
    groups=[];ambiguous=0
    for row in sorted(candidates,key=lambda z:-z['score']):
        row=dict(row);row['rle']=encode_roi(row['prob'],row['box'],threshold)
        if not cm.area(row['rle']):continue
        eligible=[]
        for i,g in enumerate(groups):
            if row['source'] in g['sources']:continue
            overlap=float(cm.iou([row['rle']],[g['anchor']],[0])[0,0])
            if overlap>=iou_threshold:eligible.append((overlap,i))
        eligible.sort(reverse=True)
        if len(eligible)>1 and eligible[0][0]-eligible[1][0]<ambiguity_margin:ambiguous+=1;continue
        if eligible:
            g=groups[eligible[0][1]];g['members'].append(row);g['sources'].add(row['source'])
        else:groups.append(dict(anchor=row['rle'],members=[row],sources={row['source']}))
    return groups,ambiguous

def vote(groups,total_views,min_support=2,threshold=.5):
    output=[]
    for group in groups:
        members=group['members'];support=len(group['sources'])
        if support<min_support:continue
        box=[min(z['box'][0] for z in members),min(z['box'][1] for z in members),max(z['box'][2] for z in members),max(z['box'][3] for z in members)];a,b,c,d=box
        numerator=np.zeros((d-b,c-a),np.float32);votes=np.zeros((d-b,c-a),np.uint16);denominator=sum(z['score'] for z in members)
        for z in members:
            x,y,w,h=z['box'];sl=np.s_[y-b:h-b,x-a:w-a];numerator[sl]+=z['prob']*z['score'];votes[sl]+=z['prob']>=threshold
        prob=numerator/max(denominator,1e-12)
        # Strict majority of matched views; no union of detections or GT.
        prob[votes<(support//2+1)]=0
        rle=encode_roi(prob,box,threshold)
        if not cm.area(rle):continue
        score=(sum(z['score'] for z in members)/support)*(support/total_views)**.5
        output.append(dict(box=box,prob=prob,score=score,support=support,total_views=total_views,sources=sorted(group['sources']),rle=rle))
    return sorted(output,key=lambda z:-z['score'])

def mask_nms(groups,iou_threshold=.8,max_det=100):
    kept=[]
    for g in sorted(groups,key=lambda z:-z['score']):
        if any(float(cm.iou([g['rle']],[z['rle']],[0])[0,0])>=iou_threshold for z in kept):continue
        kept.append(g)
        if len(kept)>=max_det:break
    return kept

