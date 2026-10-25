"""Unmodified upstream KS3 reference for paired kernel comparisons.

The latest user-selected experimental path is BLAS with reused FP8 operands.
It is NOT installed into either MLP/MoE control: those preserve upstream FP8.
"""
import hashlib
from pathlib import Path
from ks3_research.vendor import ks3_int8 as upstream

SOURCE_URL='https://github.com/curvedinf/k-splanifolds/blob/ea7a9c1121efe711345bd9e6cf22e69401c0c7b4/experiments/common/ks3_int8.py'
SOURCE_SHA256='becdb04749963afa8adc39812a61ec3bb14b0ad966c2fc04e1e4edd4ec1ed501'
if hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest()!=SOURCE_SHA256:
    raise RuntimeError('The user-selected KS3 kernel has been modified')
ks3_int8=upstream.ks3_int8
KS3Int8=upstream.KS3Int8
fuse_training=upstream.fuse_training
