import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import copy

# ideal N, CA, C initial coordinates
init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
init_CA = torch.zeros_like(init_N)
init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
INIT_CRDS = torch.zeros((27, 3)).float()
INIT_CRDS[:3] = torch.stack((init_N, init_CA, init_C), dim=0) # (3,3)

norm_N = init_N / (torch.norm(init_N, dim=-1, keepdim=True) + 1e-5)
norm_C = init_C / (torch.norm(init_C, dim=-1, keepdim=True) + 1e-5)
cos_ideal_NCAC = torch.sum(norm_N*norm_C, dim=-1) # cosine of ideal N-CA-C bond angle


class XYZConverter(nn.Module):
    def __init__(self):
        super(XYZConverter, self).__init__()
        torsion_indices = torch.full((22,4,4),0)
        torsion_can_flip = torch.full((22,10),False,dtype=torch.bool)
        reference_angles = torch.ones((22,3,2))
        tip_indices = torch.full((22,3), -1)
        base_indices = torch.full((22,27),0, dtype=torch.long)
        RTs_by_torsion = torch.eye(4).repeat(22,7,1,1)
        xyzs_in_base_frame = torch.ones((22,27,4))
        
        self.register_buffer("torsion_indices", torsion_indices)
        self.register_buffer("torsion_can_flip", torsion_can_flip)
        self.register_buffer("ref_angles", reference_angles)
        self.register_buffer("tip_indices", tip_indices)
        self.register_buffer("base_indices", base_indices)
        self.register_buffer("RTs_in_base_frame", RTs_by_torsion)
        self.register_buffer("xyzs_in_base_frame", xyzs_in_base_frame)
        num2aa=[
            'ALA','ARG','ASN','ASP','CYS',
            'GLN','GLU','GLY','HIS','ILE',
            'LEU','LYS','MET','PHE','PRO',
            'SER','THR','TRP','TYR','VAL',
            'UNK','MAS',
            ]

        self.aa2num= {x:i for i,x in enumerate(num2aa)}
    def compute_all_atom(self, seq, xyz, alphas, non_ideal=True, use_H=True):
        B,L = xyz.shape[:2]

        Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], non_ideal=non_ideal)

        RTF0 = torch.eye(4).repeat(B,L,1,1).to(device=Rs.device)

        # bb
        RTF0[:,:,:3,:3] = Rs
        RTF0[:,:,:3,3] = Ts

        # omega
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq,0,:], make_rotX(alphas[:,:,0,:]))

        # phi
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,1,:], make_rotX(alphas[:,:,1,:]))

        # psi
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,2,:], make_rotX(alphas[:,:,2,:]))

        # CB bend
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5*(basexyzs[:,:,2,:3]+basexyzs[:,:,0,:3])
        CAr = (basexyzs[:,:,1,:3])
        CBr = (basexyzs[:,:,4,:3])
        CBrotaxis1 = (CBr-CAr).cross(NCr-CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True)+1e-8
        
        # CB twist
        NCp = basexyzs[:,:,2,:3] - basexyzs[:,:,0,:3]
        NCpp = NCp - torch.sum(NCp*NCr, dim=-1, keepdim=True)/ torch.sum(NCr*NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr-CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True)+1e-8
        
        CBrot1 = make_rot_axis(alphas[:,:,7,:], CBrotaxis1 )
        CBrot2 = make_rot_axis(alphas[:,:,8,:], CBrotaxis2 )
        
        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, CBrot1,CBrot2)
        
        # chi1 + CG bend
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim', 
            RTF8, 
            self.RTs_in_base_frame[seq,3,:], 
            make_rotX(alphas[:,:,3,:]), 
            make_rotZ(alphas[:,:,9,:]))

        # chi2
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF4, self.RTs_in_base_frame[seq,4,:],make_rotX(alphas[:,:,4,:]))

        # chi3
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF5,self.RTs_in_base_frame[seq,5,:],make_rotX(alphas[:,:,5,:]))

        # chi4
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF6,self.RTs_in_base_frame[seq,6,:],make_rotX(alphas[:,:,6,:]))

        RTframes = torch.stack((
            RTF0,RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7,RTF8
        ),dim=2)

        xyzs = torch.einsum(
            'brtij,brtj->brti', 
            RTframes.gather(2,self.base_indices[seq][...,None,None].repeat(1,1,1,4,4)), basexyzs
        )

        if use_H:
            return RTframes, xyzs[...,:3]
        else:
            return RTframes, xyzs[...,:14,:3]

    def get_tor_mask(self, seq, mask_in=None): 
        B,L = seq.shape[:2]
        tors_mask = torch.ones((B,L,10), dtype=torch.bool, device=seq.device)
        tors_mask[...,3:7] = self.torsion_indices[seq,:,-1] > 0
        tors_mask[:,0,1] = False
        tors_mask[:,-1,0] = False

        # mask for additional angles
        tors_mask[:,:,7] = seq!=self.aa2num['GLY']
        tors_mask[:,:,8] = seq!=self.aa2num['GLY']
        tors_mask[:,:,9] = torch.logical_and( seq!=self.aa2num['GLY'], seq!=self.aa2num['ALA'] )
        tors_mask[:,:,9] = torch.logical_and( tors_mask[:,:,9], seq!=self.aa2num['UNK'] )
        tors_mask[:,:,9] = torch.logical_and( tors_mask[:,:,9], seq!=self.aa2num['MAS'] )

        if mask_in != None:
            # mask for missing atoms
            # chis
            ti0 = torch.gather(mask_in,2,self.torsion_indices[seq,:,0])
            ti1 = torch.gather(mask_in,2,self.torsion_indices[seq,:,1])
            ti2 = torch.gather(mask_in,2,self.torsion_indices[seq,:,2])
            ti3 = torch.gather(mask_in,2,self.torsion_indices[seq,:,3])
            #is_valid = torch.stack((ti0, ti1, ti2, ti3), dim=-2).all(dim=-1) # bug.... (2023 Feb 24 fixed)
            is_valid = torch.stack((ti0, ti1, ti2, ti3), dim=-1).all(dim=-1)
            tors_mask[...,3:7] = torch.logical_and(tors_mask[...,3:7], is_valid)
            tors_mask[:,:,7] = torch.logical_and(tors_mask[:,:,7], mask_in[:,:,4]) # CB exist?
            tors_mask[:,:,8] = torch.logical_and(tors_mask[:,:,8], mask_in[:,:,4]) # CB exist?
            tors_mask[:,:,9] = torch.logical_and(tors_mask[:,:,9], mask_in[:,:,5]) # XG exist?

        return tors_mask

    def get_torsions(self, xyz_in, seq, mask_in=None):
        B,L = xyz_in.shape[:2]
        
        tors_mask = self.get_tor_mask(seq, mask_in)
        
        # torsions to restrain to 0 or 180degree
        tors_planar = torch.zeros((B, L, 10), dtype=torch.bool, device=xyz_in.device)
        tors_planar[:,:,5] = (seq == self.aa2num['TYR']) # TYR chi 3 should be planar

        # idealize given xyz coordinates before computing torsion angles
        xyz = xyz_in.clone()
        Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:])
        Nideal = torch.tensor([-0.5272, 1.3593, 0.000], device=xyz_in.device)
        Cideal = torch.tensor([1.5233, 0.000, 0.000], device=xyz_in.device)
        xyz[...,0,:] = torch.einsum('brij,j->bri', Rs, Nideal) + Ts
        xyz[...,2,:] = torch.einsum('brij,j->bri', Rs, Cideal) + Ts

        torsions = torch.zeros( (B,L,10,2), device=xyz.device )
        # avoid undefined angles for H generation
        torsions[:,0,1,0] = 1.0
        torsions[:,-1,0,0] = 1.0

        # omega
        torsions[:,:-1,0,:] = th_dih(xyz[:,:-1,1,:],xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:])
        # phi
        torsions[:,1:,1,:] = th_dih(xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:],xyz[:,1:,2,:])
        # psi
        torsions[:,:,2,:] = -1 * th_dih(xyz[:,:,0,:],xyz[:,:,1,:],xyz[:,:,2,:],xyz[:,:,3,:])

        # chis
        ti0 = torch.gather(xyz,2,self.torsion_indices[seq,:,0,None].repeat(1,1,1,3))
        ti1 = torch.gather(xyz,2,self.torsion_indices[seq,:,1,None].repeat(1,1,1,3))
        ti2 = torch.gather(xyz,2,self.torsion_indices[seq,:,2,None].repeat(1,1,1,3))
        ti3 = torch.gather(xyz,2,self.torsion_indices[seq,:,3,None].repeat(1,1,1,3))
        torsions[:,:,3:7,:] = th_dih(ti0,ti1,ti2,ti3)
        
        # CB bend
        NC = 0.5*( xyz[:,:,0,:3] + xyz[:,:,2,:3] )
        CA = xyz[:,:,1,:3]
        CB = xyz[:,:,4,:3]
        t = th_ang_v(CB-CA,NC-CA)
        t0 = self.ref_angles[seq][...,0,:]
        torsions[:,:,7,:] = torch.stack( 
            (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
            dim=-1 )
        
        # CB twist
        NCCA = NC-CA
        NCp = xyz[:,:,2,:3] - xyz[:,:,0,:3]
        NCpp = NCp - torch.sum(NCp*NCCA, dim=-1, keepdim=True)/ torch.sum(NCCA*NCCA, dim=-1, keepdim=True) * NCCA
        t = th_ang_v(CB-CA,NCpp)
        t0 = self.ref_angles[seq][...,1,:]
        torsions[:,:,8,:] = torch.stack( 
            (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
            dim=-1 )

        # CG bend
        CG = xyz[:,:,5,:3]
        t = th_ang_v(CG-CB,CA-CB)
        t0 = self.ref_angles[seq][...,2,:]
        torsions[:,:,9,:] = torch.stack( 
            (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
            dim=-1 )
        
        tors_mask *= (~torch.isnan(torsions[...,0]))
        tors_mask *= (~torch.isnan(torsions[...,1]))
        torsions = torch.nan_to_num(torsions)

        # alt chis
        torsions_alt = torsions.clone()
        torsions_alt[self.torsion_can_flip[seq,:]] *= -1

        return torsions, torsions_alt, tors_mask, tors_planar

def th_dih_v(ab,bc,cd):
    def th_cross(a,b):
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
    n1 = th_N( th_cross(ab,bc) )
    n2 = th_N( th_cross(bc,cd) )
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)
    cos_angle = (n1*n2).sum(-1)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih
    
def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    #N, Ca, C - [B,L, 3]
    #R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B,L = N.shape[:2]
    
    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix
    
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.clamp( torch.sum(e1*v2, dim=-1), min=-1.0, max=1.0) # cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
        Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel
    
        R = torch.einsum('blij,bljk->blik', R,Rp)

    return R, Ca

def th_dih(a,b,c,d):
    return th_dih_v(a-b,b-c,c-d)

def th_ang_v(ab,bc,eps:float=1e-8):
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)
    ab, bc = th_N(ab),th_N(bc)
    cos_angle = torch.clamp( (ab*bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih