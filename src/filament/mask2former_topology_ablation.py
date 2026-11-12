"""OPT030 matched-instance boundary/topology losses and staged augmentation."""
import types
import torch
import torch.nn.functional as F

def erode(p):
    return torch.minimum(-F.max_pool2d(-p,(3,1),1,(1,0)),-F.max_pool2d(-p,(1,3),1,(0,1)))
def skeleton(p,iterations=20):
    opened=F.max_pool2d(erode(p),3,1,1);s=F.relu(p-opened)
    for _ in range(iterations):
        p=erode(p);d=F.relu(p-F.max_pool2d(erode(p),3,1,1));s=s+F.relu(d-s*d)
    return s

def extra_losses(logits,gt,compute_boundary=True,compute_topology=True):
    # Native mask-logit grid; per-instance targets, never union labels or enforce one component.
    # Chunk-safe weighted BCE and rematerialized soft skeleton limit peak probe/train memory.
    zero=logits.new_zeros((),dtype=torch.float32)
    boundary=zero;topology=zero
    if compute_boundary:
        g=gt.float()[:,None]
        if g.shape[-2:]!=logits.shape[-2:]:g=F.interpolate(g,size=logits.shape[-2:],mode='nearest')
        outer=F.max_pool2d(g,3,1,1);inner=-F.max_pool2d(-g,3,1,1);ring=(outer-inner).clamp(0,1)
        weight=1+4*ring
        boundary=F.binary_cross_entropy_with_logits(logits.float()[:,None],g,weight=weight,reduction='sum')/weight.sum().clamp_min(1)
    if compute_topology:
        g=gt.float()
        if g.shape[-2:]!=logits.shape[-2:]:g=F.interpolate(g[:,None],size=logits.shape[-2:],mode='nearest')[:,0]
        valid=g.flatten(1).sum(1)>0
        if valid.any():
            from torch.utils.checkpoint import checkpoint
            def one_instance(logit,target):
                p=logit.float().sigmoid()[None,None];target=target.float()[None,None]
                sp=skeleton(p,20)
                with torch.no_grad():sg=skeleton(target,20)
                precision=(sp*target).sum()/(sp.sum()+1e-6)
                recall=(sg*p).sum()/(sg.sum()+1e-6)
                return 1-(2*precision*recall+1e-6)/(precision+recall+1e-6)
            values=[]
            for i in torch.where(valid)[0].tolist():
                if torch.is_grad_enabled() and logits.requires_grad:
                    values.append(checkpoint(one_instance,logits[i],g[i],use_reentrant=False,preserve_rng_state=False))
                else:
                    values.append(one_instance(logits[i],g[i]))
            topology=torch.stack(values).mean()
    return boundary,topology

def install(bridge,arm,boundary_weight=.2,topology_weight=.1,phase='train'):
    c=bridge.model.criterion;original=c.forward;c.campaign_epoch=0
    c.weight_dict['loss_opt030_boundary']=boundary_weight
    c.weight_dict['loss_opt030_topology']=topology_weight
    active=arm in ('C','D','E','F_BOUNDARY','F_TOPOLOGY','F_BOTH')
    if active:
        def capture_matching(module,inputs,indices):
            if getattr(c,'_opt030_capture',False) and c._opt030_matching is None:c._opt030_matching=indices
        c.matcher.register_forward_hook(capture_matching)
    def wrapped(self,outputs,targets,*args,**kwargs):
        if active:self._opt030_capture=True;self._opt030_matching=None
        try:result=original(outputs,targets,*args,**kwargs)
        finally:
            if active:self._opt030_capture=False
        if arm not in ('C','D','E','F_BOUNDARY','F_TOPOLOGY','F_BOTH'):return result
        indices=self._opt030_matching
        if indices is None:raise RuntimeError('Native final-decoder Hungarian indices were not captured')
        terms=[];vanished=0
        for b,(src,dst) in enumerate(indices):
            if len(src):
                gt=targets[b]["masks"][dst]
                reduced=F.interpolate(gt[:,None].float(),size=outputs["pred_masks"].shape[-2:],mode="nearest")[:,0]
                vanished+=int((~(reduced>0).flatten(1).any(1)).sum())
                need_boundary=arm in ('C','E','F_BOUNDARY','F_BOTH')
                need_topology=arm in ('D','E','F_TOPOLOGY','F_BOTH')
                terms.append(extra_losses(outputs["pred_masks"][b,src],gt,need_boundary,need_topology))
        self.opt030_target_audit=dict(matched_instances=sum(len(src) for src,dst in indices),native_logit_size=list(outputs['pred_masks'].shape[-2:]),targets_vanished_at_native_grid=vanished)
        zero=outputs['pred_masks'].float().sum()*0
        boundary=torch.stack([x[0] for x in terms]).mean() if terms else zero
        topology=torch.stack([x[1] for x in terms]).mean() if terms else zero
        ramp=1. if phase!='train' else min(1.,max(0.,(self.campaign_epoch-5)/15.))
        if arm in ('C','E','F_BOUNDARY','F_BOTH'):result['loss_opt030_boundary']=boundary*ramp
        if arm in ('D','E','F_TOPOLOGY','F_BOTH'):result['loss_opt030_topology']=topology*ramp
        return result
    c.forward=types.MethodType(wrapped,c)

class AugmentedDatasetMixin:
    def __getitem__(self,index):
        row=super().__getitem__(index)
        if not self.training or not getattr(self,'opt030_augment',False):return row
        epoch=getattr(self,'campaign_epoch',0);strength=1. if epoch<40 else (.5 if epoch<65 else .15)
        # Existing native horizontal flip retained. These are synchronized image/mask transforms.
        image=row['image'];instances=row['instances'];masks=instances.gt_masks.tensor
        if torch.rand(())<.5:image=image.flip(-2);masks=masks.flip(-2)
        k=int(torch.randint(4,()).item());image=torch.rot90(image,k,(-2,-1));masks=torch.rot90(masks,k,(-2,-1))
        if torch.rand(())<.8*strength:
            contrast=1+(torch.rand(()).item()*2-1)*.15*strength
            gamma=1+(torch.rand(()).item()*2-1)*.15*strength
            shift=(torch.rand(()).item()*2-1)*.05*255*strength
            image=((image/255).clamp(0,1).pow(gamma)*255*contrast+shift).clamp(0,255)
        if torch.rand(())<.2*strength:image=(image+torch.randn_like(image)*2*strength).clamp(0,255)
        if torch.rand(())<.15*strength:image=F.avg_pool2d(image[None],3,1,1)[0]
        from detectron2.structures import BitMasks
        instances.gt_masks=BitMasks(masks);instances.gt_boxes=instances.gt_masks.get_bounding_boxes();row['image']=image.contiguous();return row
