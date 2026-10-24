"""Version-checked loader boundaries and one pinned attention metadata upload.

Only host-derived integer layout data is cached. Attention values, masks,
arithmetic, grouping and gradients retain the verified implementation.
"""
import numpy as np
import torch
from mi300.perf.pinned_batches import PinnedBatchStaging
from control393.bucket_training import metadata_plan,metadata_from_payload,shape_groups


def _version(tensor):
    try:return tensor._version
    except RuntimeError:return None  # Inference tensors cannot validate a cache.


def remember_boundaries(tensor,boundaries):
    """Attach the exact host values used to create an otherwise fresh tensor."""
    version=_version(tensor)
    if version is not None:
        tensor._control393_boundaries=(version,tuple(int(x) for x in boundaries))
    return tensor


def cached_boundaries(tensor):
    saved=getattr(tensor,'_control393_boundaries',None)
    return saved[1] if saved is not None and saved[0]==_version(tensor) else None


def host_boundaries(tensor):
    saved=cached_boundaries(tensor)
    return saved if saved is not None else tuple(tensor.detach().cpu().tolist())


def paired_boundaries(tensor):
    result=2*tensor
    saved=cached_boundaries(tensor)
    if saved is not None:remember_boundaries(result,(2*x for x in saved))
    return result


class HostBoundaryStaging(PinnedBatchStaging):
    def upload(self,inputs,targets,cum_seqlens,ngram_ids):
        uploaded=super().upload(inputs,targets,cum_seqlens,ngram_ids)
        remember_boundaries(uploaded[2],cum_seqlens.tolist())
        return uploaded


def batched_group_metadata(device,chunks,offsets,window,heads,compact=False):
    plans=[metadata_plan(chunks,ids,offsets) for ids in shape_groups(chunks)]
    if not plans:return
    host=torch.from_numpy(np.concatenate([plan[4] for plan in plans]))
    if torch.device(device).type=='cuda':host=host.pin_memory()
    # PyTorch's pinned allocator records the DMA's lifetime before reusing
    # host storage. Integer payloads are read only for this attention call.
    uploaded=host.to(device,non_blocking=True)
    for plan,payload in zip(plans,uploaded.split([p[4].size for p in plans])):
        yield metadata_from_payload(plan,payload,window,heads,compact)
