#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.mdamath import triclinic_vectors
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "MDAnalysis is required. Install with: pip install MDAnalysis"
    ) from exc


AVOGADRO = 6.02214076e23


@dataclass
class StaticInputConfig:
    """
    Human-input configuration.
    Edit this class instance in `main()` before running.
    """

    tpr_path: str
    # itp files are kept for provenance only (not parsed by this script).
    itp_paths: List[str] = field(default_factory=list)
    # Optional: needed only if you want Mc_effective [g/mol].
    density_g_cm3: Optional[float] = None

    # Required chemistry rules.
    residue_role: Dict[str, str] = field(default_factory=dict)
    reactive_atom_names: Dict[str, Set[str]] = field(default_factory=dict)
    reactive_atom_types: Dict[str, Set[str]] = field(default_factory=dict)
    allowed_role_pairs: Set[frozenset[str]] = field(default_factory=set)

    # Rule switches.
    require_inter_residue_bond: bool = True
    short_branch_max_distance: int = 2

    # Outputs.
    out_csv: Path = Path("static_crosslink_features.csv")
    out_json: Path = Path("static_crosslink_features.json")


class CrosslinkRuleBook:
    """
    Rule container without external parser usage.
    """

    def __init__(self, cfg: StaticInputConfig):
        self.residue_role = cfg.residue_role
        self.reactive_atom_names = cfg.reactive_atom_names
        self.reactive_atom_types = cfg.reactive_atom_types
        self.allowed_role_pairs = cfg.allowed_role_pairs
        self.require_inter_residue_bond = cfg.require_inter_residue_bond

    def role_of(self, atom) -> Optional[str]:
        return self.residue_role.get(atom.resname)

    def is_reactive_atom(self, atom) -> bool:
        role = self.role_of(atom)
        if role is None:
            return False
        names = self.reactive_atom_names.get(role, set())
        types = self.reactive_atom_types.get(role, set())
        by_name = atom.name in names if names else False
        # Some topologies may not expose atom type; guard with getattr.
        atom_type = getattr(atom, "type", None)
        by_type = atom_type in types if types else False
        return by_name or by_type

    def is_crosslink_bond(self, atom_i, atom_j) -> bool:
        role_i = self.role_of(atom_i)
        role_j = self.role_of(atom_j)
        if role_i is None or role_j is None:
            return False
        if self.require_inter_residue_bond and atom_i.resindex == atom_j.resindex:
            return False
        return frozenset((role_i, role_j)) in self.allowed_role_pairs


class StaticNetworkBuilder:
    """
    Build static reactive-site graph from a single topology snapshot.
    """

    def __init__(self, universe: mda.Universe, rules: CrosslinkRuleBook):
        self.universe = universe
        self.rules = rules

    def build_reactive_graph(self) -> nx.Graph:
        g = nx.Graph()
        reactive_idx: Set[int] = set()
        atoms = self.universe.atoms

        for atom in atoms:
            if not self.rules.is_reactive_atom(atom):
                continue
            reactive_idx.add(atom.index)
            g.add_node(
                atom.index,
                role=self.rules.role_of(atom),
                resname=atom.resname,
                resid=int(atom.resid),
                atom_name=atom.name,
            )

        for bond in atoms.bonds:
            a0, a1 = bond.atoms
            if a0.index not in reactive_idx or a1.index not in reactive_idx:
                continue
            if not self.rules.is_crosslink_bond(a0, a1):
                continue
            g.add_edge(a0.index, a1.index)

        return g


class EffectiveNetworkReducer:
    """
    Create effective network and dangling metrics.
    """

    def __init__(self, short_branch_max_distance: int = 2):
        self.short_branch_max_distance = short_branch_max_distance

    @staticmethod
    def largest_component(g: nx.Graph) -> nx.Graph:
        if g.number_of_nodes() == 0:
            return g.copy()
        comp = max(nx.connected_components(g), key=len)
        return g.subgraph(comp).copy()

    @staticmethod
    def prune_leaves_iter(g: nx.Graph) -> Tuple[nx.Graph, Set[int]]:
        h = g.copy()
        removed: Set[int] = set()
        while True:
            leaves = [n for n, d in h.degree() if d <= 1]
            if not leaves:
                break
            h.remove_nodes_from(leaves)
            removed.update(leaves)
        return h, removed

    def prune_short_branches_once(self, g: nx.Graph) -> Tuple[nx.Graph, Set[int]]:
        """
        Remove short branches from current graph:
        junction(deg>=3) -> ... -> leaf(deg=1), within max distance.
        """
        if g.number_of_nodes() == 0:
            return g.copy(), set()
        deg0 = dict(g.degree())
        removed: Set[int] = set()
        h = g.copy()

        for j, dj in deg0.items():
            if dj < 3:
                continue
            for nbr in list(g.neighbors(j)):
                path = [nbr]
                prev = j
                cur = nbr
                dist = 1
                if deg0[cur] == 1 and dist <= self.short_branch_max_distance:
                    removed.update(path)
                    continue
                if deg0[cur] >= 3:
                    continue
                while dist < self.short_branch_max_distance:
                    nxts = [x for x in g.neighbors(cur) if x not in {prev, j}]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    path.append(nxt)
                    prev, cur = cur, nxt
                    dist += 1
                    if deg0[cur] == 1:
                        removed.update(path)
                        break
                    if deg0[cur] >= 3:
                        break

        if removed:
            h.remove_nodes_from(removed)
        return h, removed

    def effective_graph(self, g: nx.Graph) -> Tuple[nx.Graph, Dict[str, int]]:
        h = self.largest_component(g)
        h1, removed_leaf = self.prune_leaves_iter(h)
        h2, removed_short = self.prune_short_branches_once(h1)
        stats = {
            "removed_leaf_nodes": len(removed_leaf),
            "removed_short_branch_nodes": len(removed_short),
            "removed_total": len(removed_leaf | removed_short),
        }
        return h2, stats


class StaticFeatureCalculator:
    def __init__(self, density_g_cm3: Optional[float] = None):
        self.density_g_cm3 = density_g_cm3

    @staticmethod
    def _box_volume_angstrom3(universe: mda.Universe) -> Optional[float]:
        dim = universe.trajectory.ts.dimensions
        if dim is None or len(dim) < 6 or np.any(np.asarray(dim[:3]) <= 0):
            return None
        # dimensions are [lx, ly, lz, alpha, beta, gamma]
        v = triclinic_vectors(dim)
        vol = abs(float(np.linalg.det(v)))
        return vol if vol > 0 else None

    def calculate(
        self,
        g: nx.Graph,
        g_eff: nx.Graph,
        reducer_stats: Dict[str, int],
        universe: mda.Universe,
    ) -> Dict[str, float]:
        n = g.number_of_nodes()
        m = g.number_of_edges()
        deg = [d for _, d in g.degree()]
        comp_count = nx.number_connected_components(g) if n else 0
        lcc_n = len(max(nx.connected_components(g), key=len)) if n else 0

        mono_n = sum(1 for _, dct in g.nodes(data=True) if dct.get("role") == "mono_epoxy")
        di_n = sum(1 for _, dct in g.nodes(data=True) if dct.get("role") == "di_epoxy")
        amine_n = sum(1 for _, dct in g.nodes(data=True) if dct.get("role") == "amine")

        leaf_n = sum(1 for x in deg if x == 1)
        junction_n = sum(1 for x in dict(g_eff.degree()).values() if x >= 3)
        dangling_frac = (reducer_stats["removed_total"] / n) if n else 0.0

        vol_a3 = self._box_volume_angstrom3(universe)
        if vol_a3 is None:
            nu_eff_nm3 = None
            nu_eff_molm3 = None
        else:
            sites_per_a3 = junction_n / vol_a3
            nu_eff_nm3 = sites_per_a3 * 1000.0  # 1 nm^3 = 1000 A^3
            nu_eff_molm3 = (sites_per_a3 * 1e30) / AVOGADRO

        if self.density_g_cm3 is not None and nu_eff_molm3 and nu_eff_molm3 > 0:
            # mol/m^3 -> mol/cm^3
            nu_eff_molcm3 = nu_eff_molm3 / 1e6
            mc_effective = self.density_g_cm3 / nu_eff_molcm3
        else:
            mc_effective = None

        out: Dict[str, float] = {
            "n_reactive_sites": float(n),
            "n_crosslink_bonds": float(m),
            "mono_func_fraction": float(mono_n / n) if n else 0.0,
            "di_func_fraction": float(di_n / n) if n else 0.0,
            "amine_fraction": float(amine_n / n) if n else 0.0,
            "n_components": float(comp_count),
            "largest_component_fraction": float(lcc_n / n) if n else 0.0,
            "leaf_fraction": float(leaf_n / n) if n else 0.0,
            "dangling_chain_fraction": float(dangling_frac),
            "effective_junction_count": float(junction_n),
            "effective_crosslink_density_nm3": float(nu_eff_nm3) if nu_eff_nm3 is not None else np.nan,
            "effective_crosslink_density_mol_m3": float(nu_eff_molm3) if nu_eff_molm3 is not None else np.nan,
            "Mc_effective_g_mol": float(mc_effective) if mc_effective is not None else np.nan,
            "cyclomatic_number_full": float(m - n + comp_count),
            "cyclomatic_number_effective": float(
                g_eff.number_of_edges() - g_eff.number_of_nodes() + nx.number_connected_components(g_eff)
            )
            if g_eff.number_of_nodes()
            else 0.0,
            "volume_angstrom3": float(vol_a3) if vol_a3 is not None else np.nan,
        }
        return out


class StaticFeatureExporter:
    @staticmethod
    def write_csv(path: Path, row: Dict[str, float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def write_json(path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def validate_config(cfg: StaticInputConfig, u: mda.Universe) -> None:
    if not Path(cfg.tpr_path).exists():
        raise SystemExit(f"tpr_path not found: {cfg.tpr_path}")
    if not cfg.residue_role:
        raise SystemExit("residue_role is empty.")
    if not cfg.allowed_role_pairs:
        raise SystemExit("allowed_role_pairs is empty.")

    observed_res = sorted(set(u.atoms.resnames))
    unknown = [r for r in observed_res if r not in cfg.residue_role]
    if unknown:
        print(
            "[warning] Residues in topology not mapped in residue_role "
            f"(ignored in analysis): {unknown}"
        )


def main() -> None:
    # -------------------------------------------------------------------------
    # Edit this block only.
    # -------------------------------------------------------------------------
    cfg = StaticInputConfig(
        tpr_path="your_system.tpr",
        itp_paths=["epoxy.itp", "amine.itp"],  # provenance only
        density_g_cm3=None,  # set e.g. 1.15 to compute Mc_effective
        residue_role={
            # "MEOX": "mono_epoxy",
            # "DGE": "di_epoxy",
            # "DIA": "amine",
        },
        reactive_atom_names={
            "mono_epoxy": set(),
            "di_epoxy": set(),
            "amine": set(),
            # e.g. "mono_epoxy": {"C_EP"},
            #      "di_epoxy": {"C_EP1", "C_EP2"},
            #      "amine": {"N1", "N2"},
        },
        reactive_atom_types={
            "mono_epoxy": set(),
            "di_epoxy": set(),
            "amine": set(),
        },
        allowed_role_pairs={
            frozenset(("mono_epoxy", "amine")),
            frozenset(("di_epoxy", "amine")),
        },
        require_inter_residue_bond=True,
        short_branch_max_distance=2,
        out_csv=Path("static_crosslink_features.csv"),
        out_json=Path("static_crosslink_features.json"),
    )
    # -------------------------------------------------------------------------

    u = mda.Universe(cfg.tpr_path)
    validate_config(cfg, u)

    rules = CrosslinkRuleBook(cfg)
    builder = StaticNetworkBuilder(u, rules)
    reducer = EffectiveNetworkReducer(short_branch_max_distance=cfg.short_branch_max_distance)
    calculator = StaticFeatureCalculator(density_g_cm3=cfg.density_g_cm3)

    g = builder.build_reactive_graph()
    g_eff, reduce_stats = reducer.effective_graph(g)
    features = calculator.calculate(g, g_eff, reduce_stats, u)

    payload = {
        "input": {
            "tpr_path": cfg.tpr_path,
            "itp_paths": cfg.itp_paths,
            "density_g_cm3": cfg.density_g_cm3,
            "residue_role": cfg.residue_role,
            "reactive_atom_names": {k: sorted(v) for k, v in cfg.reactive_atom_names.items()},
            "reactive_atom_types": {k: sorted(v) for k, v in cfg.reactive_atom_types.items()},
            "allowed_role_pairs": [sorted(list(x)) for x in cfg.allowed_role_pairs],
            "require_inter_residue_bond": cfg.require_inter_residue_bond,
            "short_branch_max_distance": cfg.short_branch_max_distance,
        },
        "graph_stats": {
            "full_nodes": g.number_of_nodes(),
            "full_edges": g.number_of_edges(),
            "effective_nodes": g_eff.number_of_nodes(),
            "effective_edges": g_eff.number_of_edges(),
            **reduce_stats,
        },
        "features": features,
    }

    exporter = StaticFeatureExporter()
    exporter.write_csv(cfg.out_csv, features)
    exporter.write_json(cfg.out_json, payload)

    print(f"Wrote features CSV: {cfg.out_csv}")
    print(f"Wrote detail JSON: {cfg.out_json}")


if __name__ == "__main__":
    main()
