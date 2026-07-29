# Global-Theta JC69 Quartet Branch-Length Estimator

This folder is a small, self-contained extraction from `caster-brlen` for aster2 integration.
It keeps only the original thesis-style model:

- fixed rooted binary species-tree topology;
- one sequence per taxon;
- JC69 substitution only;
- one global `theta = 4 Ne mu` for the whole species tree;
- branch lengths estimated in substitution units (SU), then converted to coalescent units (CU) by
  `CU = 2 * SU / theta`.

The public entry point is:

```python
from global_theta_jc69_quartet import estimate_branch_lengths
```

or the CLI module:

```bash
python -m global_theta_jc69_quartet.global_theta_jc69_estimator \
  --species-tree species_tree_topology.newick \
  --alignment alignment.fasta \
  --output-dir out \
  --prefix estimated \
  --max-iter 20 \
  --restarts 1
```

## Files

- `global_theta_jc69_quartet/global_theta_jc69_estimator.py`
  - Main n-taxon estimator and CLI.
  - Builds edge-preserving witness quartets for every non-root branch.
  - Fits each unique quartet once.
  - Aggregates quartet SU branch estimates and quartet theta estimates back to the full tree.

- `global_theta_jc69_quartet/quartet_theta_estimator.py`
  - Optimizes a single rooted quartet under JC69 and one global theta.
  - Parameters are three internal node ages in SU plus theta.

- `global_theta_jc69_quartet/quartet_site_pattern_probability.py`
  - MSC-integrated quartet site-pattern probability engine.
  - Uses symbolic case enumeration and cached symbolic-to-numeric probability functions.

- `global_theta_jc69_quartet/msc_density.py`
  - Quartet multispecies-coalescent case enumeration and density terms.

- `global_theta_jc69_quartet/rooted_gene_trees.py`
  - Enumerates rooted labeled quartet gene trees.

- `global_theta_jc69_quartet/symbolic_substitution_models.py`
  - JC69 transition probabilities only.

- `global_theta_jc69_quartet/site_pattern_counts.py`
  - FASTA/count parsing and JC69 site-pattern canonicalization.

- `global_theta_jc69_quartet/alignment_cache.py`
  - Loads alignments once and caches quartet site-pattern summaries.

- `global_theta_jc69_quartet/topology_utils.py`
  - Rooted topology parser, induced quartet extraction, and Newick formatting.

- `global_theta_jc69_quartet/annotated_species_tree.py`
  - Quartet species-tree parser used internally by the probability engine.

## Outputs

The CLI writes three files:

- `<prefix>_tree_su.newick`: estimated species tree in substitution units.
- `<prefix>_tree_cu.newick`: estimated species tree in coalescent units.
- `<prefix>_theta.txt`: estimated theta and run summary.

## Integration Notes

`generate_cpp_probabilities.py` evaluates the symbolic engine and writes the
dependency-free rooted-quartet functions used by aster2 in
`src/jc69_msc_probabilities.hpp`.

The same generated header also exposes the rooted-triplet probabilities for
`((A,B),C)` with species ages `s1 < s2`. They are exact marginals of the
unbalanced quartet `(((A,B),C),D)`:

```text
AAA = AAAA + 3 AAAC
AAC = AACA + AACC + 2 AACG
ACA = ACAA + ACAC + 2 ACAG
ACC = ACCA + ACCC + 2 ACCG
ACG = ACGA + ACGC + ACGG + ACGT
```

Sampling consistency removes the older quartet age, leaving only `s1`, `s2`,
and `theta`. The pattern positions remain A/B/C throughout; only nucleotide
states are canonically renamed.

Important algorithmic boundary: this is a quartet-composite branch-stitching estimator, not a full
n-taxon likelihood. Branch estimates are stitched from quartet fits and then summarized on the full
topology.
