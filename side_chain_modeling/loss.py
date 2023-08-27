import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

#torsion angle predictor loss
def torsionAngleLoss( alpha, alphanat, alphanat_alt, tors_mask, tors_planar, eps=1e-8 ):
    I = alpha.shape[0]
    lnat = torch.sqrt( torch.sum( torch.square(alpha), dim=-1 ) + eps )
    anorm = alpha / (lnat[...,None])

    l_tors_ij = torch.min(
            torch.sum(torch.square( anorm - alphanat[None] ),dim=-1),
            torch.sum(torch.square( anorm - alphanat_alt[None] ),dim=-1)
        )

    # l_tors = torch.sum( l_tors_ij*tors_mask[None] ) / ( torch.sqrt(torch.sum( tors_mask ))*I + eps)
    # l_norm = torch.sum( torch.abs(lnat-1.0)*tors_mask[None] ) / (torch.sqrt(torch.sum( tors_mask ))*I + eps)
    # l_planar = torch.sum( torch.abs( alpha[...,0] )*tors_planar[None] ) / (torch.sqrt(torch.sum( tors_planar ))*I + eps)
    l_tors = torch.sum( l_tors_ij*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_norm = torch.sum( torch.abs(lnat-1.0)*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_planar = torch.sum( torch.abs( alpha[...,0] )*tors_planar[None] ) / (torch.sum( tors_planar )*I + eps)

    return l_tors+0.02*l_norm+0.02*l_planar
