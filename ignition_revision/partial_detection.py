"""Explicit partial-label sampling for torchvision Faster R-CNN (eager mode).

Unknown boxes exclude negative anchors/proposals by intersection / candidate area.
Known positive assignments take precedence. Unknown regions are not zero-filled and
are not passed as positive GT. This cannot recover objects missing from tracks.
"""
import torch
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform,resize_boxes


def ignore_negative_labels(boxes,regions,labels,threshold):
    if regions is None or regions.numel()==0 or boxes.numel()==0:return labels
    low=torch.maximum(boxes[:,None,:2],regions[None,:,:2]);high=torch.minimum(boxes[:,None,2:],regions[None,:,2:])
    inter=(high-low).clamp(min=0).prod(dim=2)
    area=(boxes[:,2:]-boxes[:,:2]).clamp(min=0).prod(dim=1).clamp(min=1e-8)
    overlap=(inter/area[:,None]).amax(dim=1)
    result=labels.clone();result[(labels==0)&(overlap>=threshold)]=-1
    return result


class IgnoreTransform(GeneralizedRCNNTransform):
    def resize(self,image,target=None):
        if getattr(self,'keep_original_resolution',False):return image,target
        old=image.shape[-2:];regions=target.get('ignore_boxes') if target is not None else None
        image,target=super().resize(image,target)
        if regions is not None:target['ignore_boxes']=resize_boxes(regions,old,image.shape[-2:])
        return image,target


class IgnoreRPN(RegionProposalNetwork):
    def assign_targets_to_anchors(self,anchors,targets):
        labels,matched=super().assign_targets_to_anchors(anchors,targets)
        labels=[ignore_negative_labels(a,t.get('ignore_boxes'),y,self.ignore_ioa) for a,t,y in zip(anchors,targets,labels)]
        return labels,matched


class IgnoreRoIHeads(RoIHeads):
    def select_training_samples(self,proposals,targets):
        # One model owner per training process; state exists only during this call.
        self._sample_ignore_regions=[t.get('ignore_boxes') for t in targets]
        try:return super().select_training_samples(proposals,targets)
        finally:self._sample_ignore_regions=None
    def assign_targets_to_proposals(self,proposals,gt_boxes,gt_labels):
        matched,labels=super().assign_targets_to_proposals(proposals,gt_boxes,gt_labels)
        if self._sample_ignore_regions is not None:
            labels=[ignore_negative_labels(p,g,y,self.ignore_ioa) for p,g,y in zip(proposals,self._sample_ignore_regions,labels)]
        return matched,labels


def install_partial_labels(model,threshold=.5):
    if not 0<float(threshold)<=1:raise ValueError('ignore_ioa must be in (0,1]')
    # Retain initialized modules/state_dict names; only extend their sampling methods.
    model.transform.__class__=IgnoreTransform
    model.rpn.__class__=IgnoreRPN;model.rpn.ignore_ioa=float(threshold)
    model.roi_heads.__class__=IgnoreRoIHeads;model.roi_heads.ignore_ioa=float(threshold);model.roi_heads._sample_ignore_regions=None
    return model
