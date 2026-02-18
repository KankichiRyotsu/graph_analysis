# Static Crosslink Parameterization: Input Guide

Target script:
- `/Users/sugimoriryouta/research/graph_theory/static_crosslink_parameterizer.py`

This workflow does not parse ITP files automatically.
You define rules directly in the class config block.

## 1. Required input items

- `tpr_path`
  - Source: your GROMACS topology file used for the cured system.
- `residue_role`
  - Mapping: `resname -> mono_epoxy | di_epoxy | amine`
  - Source:
    1. Resin recipe (which molecule is mono/di/amine)
    2. Residue names in topology (`tpr`) and ITP naming convention
- `reactive_atom_names` or `reactive_atom_types`
  - Mapping: role -> reactive atom names/types
  - Source:
    1. ITP `[ atoms ]` atom names/types
    2. Reaction mechanism definition (epoxy-side reactive C, amine-side reactive N)
- `allowed_role_pairs`
  - Example:
    - `("mono_epoxy", "amine")`
    - `("di_epoxy", "amine")`
  - Source: chemistry of allowed crosslink reactions.

## 2. Optional items

- `density_g_cm3`
  - Needed only for `Mc_effective_g_mol`.
  - Source: experiment or MD-equilibrated density.
- `short_branch_max_distance`
  - Controls dangling short-branch removal (default 2).
  - Source: analysis policy choice.

## 3. Suggested preparation steps

1. List all residue names in the cured `tpr`.
2. Assign each relevant residue to `mono_epoxy/di_epoxy/amine`.
3. For each role, specify reactive atom names from ITP.
4. Run script and inspect warnings for unmapped residues.
5. Validate counts against expected stoichiometry.

## 4. Output definitions

- `effective_crosslink_density_nm3`:
  - `(# junction nodes in effective graph) / box volume [nm^3]`
  - junction node = node degree >= 3 after effective-network reduction.
- `dangling_chain_fraction`:
  - fraction of nodes removed during leaf/short-branch pruning.
