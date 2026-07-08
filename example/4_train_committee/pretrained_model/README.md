# Pretrained committee model

The committee neural-network potential used in this work, plus the datasets it
was trained on. Drop-in for stage 5 (`5_qmml_run/`) if you want to skip the
data-generation + training pipeline (stages 1–4) and go straight to a QM/ML run.

All quantities follow the RuNNer/n2p2 **atomic-unit** convention: positions and
cell in **Bohr**, energy in **Hartree**, forces in **Hartree/Bohr**. The target
is the **non-electrostatic (short-range) potential** — i.e. the charge-decoupled
reference (`input-SR − coulomb`) produced by stage 3; the long-range
electrostatics are re-added at run time by the mean-field DFT-CES coupling, not
learned here.

## `final_model/` — the committee

A committee of **7** independently trained n2p2 NNPs (`nnp-data-1/` …
`nnp-data-7/`). This is the pretrained model mainly employed in this work.

Elements (fixed order): **O H C Na Cl** (5 elements).

Each member directory contains what LAMMPS `pair_style nnp` loads:

```
nnp-data-<i>/
  input.nn            # network + symmetry-function definition
  scaling.data        # symmetry-function min/max/mean scaling
  weights.<Z>.data    # trained weights, one file per element by atomic number:
                      #   001=H  006=C  008=O  011=Na  017=Cl
```

The files at the top of `final_model/` (`input.nn`, `scaling.data`,
`weights.*.data`) are a single representative network; the committee that LAMMPS
actually reads is the `nnp-data-1..7/` set.

**Use it in stage 5** — point `base.in.lammps` at this directory:

```
pair_style nnp dir "…/pretrained_model/final_model/" showew no showewsum 10000 \
    resetew yes ...
```

This is the same layout `train_committee()` produces (its `out_dir="committee"`),
so a model you train yourself is interchangeable with this one.

## `Data/` — training datasets (n2p2 `.data`)

Non-ES training data, in RuNNer/n2p2 `.data` format (one `begin … end` block per
frame, with `lattice`, `atom` (position, element, charge, force), and `energy`):

| file | frames | role |
|---|---|---|
| `initial.data`  | 877  | initial training set for the base model |
| `initial2.data` | 1111 | additional initial training data (extends the base set) |
| `Applied_V.data`| 79   | applied-potential frames used to **fine-tune** the model |

The cell in these frames is ~23.34 × 24.25 Bohr in x, y (≈ 12.35 × 12.83 Å) — the
NaCl-in-water / graphene-electrode system of this example.

To retrain from these instead of running stages 1–3, feed them to
`train_committee(data_file=..., ...)` in `4_train_committee/run_train.py`
(concatenate the `.data` files you want, keep `elements=("O","H","C","Na","Cl")`,
and set `n_members=7` to match this committee).
