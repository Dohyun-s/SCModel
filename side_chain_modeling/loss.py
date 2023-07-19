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

    l_tors = torch.sum( l_tors_ij*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_norm = torch.sum( torch.abs(lnat-1.0)*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_planar = torch.sum( torch.abs( alpha[...,0] )*tors_planar[None] ) / (torch.sum( tors_planar )*I + eps)

    return l_tors+0.02*l_norm+0.02*l_planar



# # def supervised_chi_loss(ret, batch, value, config):
#     """Computes loss for direct chi angle supervision.

#     Jumper et al. (2021) Suppl. Alg. 27 "torsionAngleLoss"

#     Args:
#       ret: Dictionary to write outputs into, needs to contain 'loss'.
#       batch: Batch, needs to contain 'seq_mask', 'chi_mask', 'chi_angles'.
#       value: Dictionary containing structure module output, needs to contain
#         value['sidechains']['angles_sin_cos'] for angles and
#         value['sidechains']['unnormalized_angles_sin_cos'] for unnormalized
#         angles.
#       config: Configuration of loss, should contain 'chi_weight' and
#         'angle_norm_weight', 'angle_norm_weight' scales angle norm term,
#         'chi_weight' scales torsion term.
#     """
#     eps = 1e-6

#     sequence_mask = batch["seq_mask"]
#     num_res = sequence_mask.shape[0]
#     chi_mask = batch["chi_mask"].float()
#     pred_angles = value["sidechains"]["angles_sin_cos"].view(-1, num_res, 7, 2)
#     pred_angles = pred_angles[:, :, 3:]

#     residue_type_one_hot = F.one_hot(
#         batch["aatype"], residue_constants.restype_num + 1
#     ).float()[None]
#     chi_pi_periodic = torch.tensor(
#         residue_constants.chi_pi_periodic
#     ).to(residue_type_one_hot.device)

#     true_chi = batch["chi_angles"][None]
#     sin_true_chi = torch.sin(true_chi)
#     cos_true_chi = torch.cos(true_chi)
#     sin_cos_true_chi = torch.stack([sin_true_chi, cos_true_chi], dim=-1)

#     # This is -1 if chi is pi-periodic and +1 if it's 2pi-periodic
#     shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
#     sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi

#     sq_chi_error = torch.sum(torch.square(sin_cos_true_chi - pred_angles), dim=-1)
#     sq_chi_error_shifted = torch.sum(
#         torch.square(sin_cos_true_chi_shifted - pred_angles), dim=-1
#     )
#     sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

#     sq_chi_loss = utils.mask_mean(mask=chi_mask[None], value=sq_chi_error)
#     ret["chi_loss"] = sq_chi_loss
#     ret["loss"] += config.chi_weight * sq_chi_loss
#     unnormed_angles = value["sidechains"]["unnormalized_angles_sin_cos"].view(
#         -1, num_res, 7, 2
#     )
#     angle_norm = torch.sqrt(torch.sum(torch.square(unnormed_angles), dim=-1) + eps)
#     norm_error = torch.abs(angle_norm - 1.0)
#     angle_norm_loss = utils.mask_mean(
#         mask=sequence_mask[None, :, None], value=norm_error
#     )

#     ret["angle_norm_loss"] = angle_norm_loss
#     ret["loss"] += config.angle_norm_weight * angle_norm_loss



# # # def atom37_to_torsion_angles(
# #     aatype: jnp.ndarray,  # (B, N)
# #     all_atom_pos: jnp.ndarray,  # (B, N, 37, 3)
# #     all_atom_mask: jnp.ndarray,  # (B, N, 37)
# #     placeholder_for_undefined=False,
# # ) -> Dict[str, jnp.ndarray]:
# #     """Computes the 7 torsion angles (in sin, cos encoding) for each residue.

# #     The 7 torsion angles are in the order
# #     '[pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]',
# #     here pre_omega denotes the omega torsion angle between the given amino acid
# #     and the previous amino acid.

# #     Args:
# #       aatype: Amino acid type, given as array with integers.
# #       all_atom_pos: atom37 representation of all atom coordinates.
# #       all_atom_mask: atom37 representation of mask on all atom coordinates.
# #       placeholder_for_undefined: flag denoting whether to set masked torsion
# #         angles to zero.
# #     Returns:
# #       Dict containing:
# #         * 'torsion_angles_sin_cos': Array with shape (B, N, 7, 2) where the final
# #           2 dimensions denote sin and cos respectively
# #         * 'alt_torsion_angles_sin_cos': same as 'torsion_angles_sin_cos', but
# #           with the angle shifted by pi for all chi angles affected by the naming
# #           ambiguities.
# #         * 'torsion_angles_mask': Mask for which chi angles are present.
# #     """

# #     # Map aatype > 20 to 'Unknown' (20).
# #     aatype = jnp.minimum(aatype, 20)

# #     # Compute the backbone angles.
# #     num_batch, num_res = aatype.shape

# #     pad = jnp.zeros([num_batch, 1, 37, 3], jnp.float32)
# #     prev_all_atom_pos = jnp.concatenate([pad, all_atom_pos[:, :-1, :, :]], axis=1)

# #     pad = jnp.zeros([num_batch, 1, 37], jnp.float32)
# #     prev_all_atom_mask = jnp.concatenate([pad, all_atom_mask[:, :-1, :]], axis=1)

# #     # For each torsion angle collect the 4 atom positions that define this angle.
# #     # shape (B, N, atoms=4, xyz=3)
# #     pre_omega_atom_pos = jnp.concatenate(
# #         [
# #             prev_all_atom_pos[:, :, 1:3, :],  # prev CA, C
# #             all_atom_pos[:, :, 0:2, :],  # this N, CA
# #         ],
# #         axis=-2,
# #     )
# #     phi_atom_pos = jnp.concatenate(
# #         [
# #             prev_all_atom_pos[:, :, 2:3, :],  # prev C
# #             all_atom_pos[:, :, 0:3, :],  # this N, CA, C
# #         ],
# #         axis=-2,
# #     )
# #     psi_atom_pos = jnp.concatenate(
# #         [
# #             all_atom_pos[:, :, 0:3, :],  # this N, CA, C
# #             all_atom_pos[:, :, 4:5, :],  # this O
# #         ],
# #         axis=-2,
# #     )

# #     # Collect the masks from these atoms.
# #     # Shape [batch, num_res]
# #     pre_omega_mask = jnp.prod(
# #         prev_all_atom_mask[:, :, 1:3], axis=-1
# #     ) * jnp.prod(  # prev CA, C
# #         all_atom_mask[:, :, 0:2], axis=-1
# #     )  # this N, CA
# #     phi_mask = prev_all_atom_mask[:, :, 2] * jnp.prod(  # prev C
# #         all_atom_mask[:, :, 0:3], axis=-1
# #     )  # this N, CA, C
# #     psi_mask = (
# #         jnp.prod(all_atom_mask[:, :, 0:3], axis=-1)
# #         * all_atom_mask[:, :, 4]  # this N, CA, C
# #     )  # this O

# #     # Collect the atoms for the chi-angles.
# #     # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
# #     chi_atom_indices = get_chi_atom_indices()
# #     # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
# #     atom_indices = utils.batched_gather(
# #         params=chi_atom_indices, indices=aatype, axis=0, batch_dims=0
# #     )
# #     # Gather atom positions. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].
# #     chis_atom_pos = utils.batched_gather(
# #         params=all_atom_pos, indices=atom_indices, axis=-2, batch_dims=2
# #     )

# #     # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
# #     chi_angles_mask = list(residue_constants.chi_angles_mask)
# #     chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
# #     chi_angles_mask = jnp.asarray(chi_angles_mask)

# #     # Compute the chi angle mask. I.e. which chis angles exist according to the
# #     # aatype. Shape [batch, num_res, chis=4].
# #     chis_mask = utils.batched_gather(
# #         params=chi_angles_mask, indices=aatype, axis=0, batch_dims=0
# #     )

# #     # Constrain the chis_mask to those chis, where the ground truth coordinates of
# #     # all defining four atoms are available.
# #     # Gather the chi angle atoms mask. Shape: [batch, num_res, chis=4, atoms=4].
# #     chi_angle_atoms_mask = utils.batched_gather(
# #         params=all_atom_mask, indices=atom_indices, axis=-1, batch_dims=2
# #     )
# #     # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
# #     chi_angle_atoms_mask = jnp.prod(chi_angle_atoms_mask, axis=[-1])
# #     chis_mask = chis_mask * (chi_angle_atoms_mask).astype(jnp.float32)

# #     # Stack all torsion angle atom positions.
# #     # Shape (B, N, torsions=7, atoms=4, xyz=3)
# #     torsions_atom_pos = jnp.concatenate(
# #         [
# #             pre_omega_atom_pos[:, :, None, :, :],
# #             phi_atom_pos[:, :, None, :, :],
# #             psi_atom_pos[:, :, None, :, :],
# #             chis_atom_pos,
# #         ],
# #         axis=2,
# #     )

# #     # Stack up masks for all torsion angles.
# #     # shape (B, N, torsions=7)
# #     torsion_angles_mask = jnp.concatenate(
# #         [
# #             pre_omega_mask[:, :, None],
# #             phi_mask[:, :, None],
# #             psi_mask[:, :, None],
# #             chis_mask,
# #         ],
# #         axis=2,
# #     )

# #     # Create a frame from the first three atoms:
# #     # First atom: point on x-y-plane
# #     # Second atom: point on negative x-axis
# #     # Third atom: origin
# #     # r3.Rigids (B, N, torsions=7)
# #     torsion_frames = r3.rigids_from_3_points(
# #         point_on_neg_x_axis=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 1, :]),
# #         origin=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 2, :]),
# #         point_on_xy_plane=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 0, :]),
# #     )

# #     # Compute the position of the forth atom in this frame (y and z coordinate
# #     # define the chi angle)
# #     # r3.Vecs (B, N, torsions=7)
# #     forth_atom_rel_pos = r3.rigids_mul_vecs(
# #         r3.invert_rigids(torsion_frames),
# #         r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 3, :]),
# #     )

# #     # Normalize to have the sin and cos of the torsion angle.
# #     # jnp.ndarray (B, N, torsions=7, sincos=2)
# #     torsion_angles_sin_cos = jnp.stack(
# #         [forth_atom_rel_pos.z, forth_atom_rel_pos.y], axis=-1
# #     )
# #     torsion_angles_sin_cos /= jnp.sqrt(
# #         jnp.sum(jnp.square(torsion_angles_sin_cos), axis=-1, keepdims=True) + 1e-8
# #     )

# #     # Mirror psi, because we computed it from the Oxygen-atom.
# #     torsion_angles_sin_cos *= jnp.asarray([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0])[
# #         None, None, :, None
# #     ]

# #     # Create alternative angles for ambiguous atom names.
# #     chi_is_ambiguous = utils.batched_gather(
# #         jnp.asarray(residue_constants.chi_pi_periodic), aatype
# #     )
# #     mirror_torsion_angles = jnp.concatenate(
# #         [jnp.ones([num_batch, num_res, 3]), 1.0 - 2.0 * chi_is_ambiguous], axis=-1
# #     )
# #     alt_torsion_angles_sin_cos = (
# #         torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None]
# #     )

# #     if placeholder_for_undefined:
# #         # Add placeholder torsions in place of undefined torsion angles
# #         # (e.g. N-terminus pre-omega)
# #         placeholder_torsions = jnp.stack(
# #             [
# #                 jnp.ones(torsion_angles_sin_cos.shape[:-1]),
# #                 jnp.zeros(torsion_angles_sin_cos.shape[:-1]),
# #             ],
# #             axis=-1,
# #         )
# #         torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[
# #             ..., None
# #         ] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
# #         alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
# #             ..., None
# #         ] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

# #     return {
# #         "torsion_angles_sin_cos": torsion_angles_sin_cos,  # (B, N, 7, 2)
# #         "alt_torsion_angles_sin_cos": alt_torsion_angles_sin_cos,  # (B, N, 7, 2)
# #         "torsion_angles_mask": torsion_angles_mask,  # (B, N, 7)
# #     }