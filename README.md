## The Phylogenetic Occupancy Model

Phylogenetic occupancy models are probabilistic models that combine aspects of ecological occupancy models and evolutionary models to impute gene presence/absence in incomplete genomic data. 

The model utilizes an already-inferred phylogeny to construct a belief network, which is then fed into an occupancy emission model. This approach simultaneously estimates:
* Ancestral genomic content.
* Genome completeness.
* Probability of gene presence at each gene in each genome.
* Multiple-imputation of gene presence/absence.

## Installation

To install the model using `pip`, execute the following on your machine:
```
pip install git+https://github.com/Wesley-DeMontigny/Phylogenetic-Occupancy-Model.git
```

## Usage

After installation, the model can be run from the command line using the `phylo_occupancy` module:

```bash
python -m phylo_occupancy [-h] -t TREE -d DATA [-c CORE_GENES] [-g COMPLETE_GENOMES] [-o TIP_PROBABILITIES] [-m TIP_MAP] [-r RECONSTRUCT_ROOT] [-a ANCESTRAL_RECONSTRUCTION] [-i INFERRED_COMPLETENESS] [-n ANNOTATED_TREE]
```

### Command Line Arguments

| Argument | Name | Required | Description |
| :--- | :--- | :--- | :--- |
| `-t` | Tree | **Yes** | Rooted phylogeny in Newick format. *Note: Avoid special characters in tip names to prevent parsing errors with the `ete3` package.* |
| `-d` | Data | **Yes** | Binary `.tsv` table of orthogroup presence/absence indicators (Columns = genomes, Rows = orthogroups). An example is available at `/example/eggnog_ortholog_table.tsv`. |
| `-c` | Core Genes | No | Single-column `.tsv` of indices for genes expected to be core. *(Note: In practice, specifying this rarely changes inferences significantly).* |
| `-g` | Complete Genomes | No | Single-column `.tsv` of indices for genomes expected to be complete. *(Note: Similar to `-c`, this rarely impacts inferences).* |
| `-o` | Tip Probabilities | No | Output `.tsv` filename for the probability of gene presence at each of the tips (genomes). |
| `-m` | Tip MAP | No | Output `.tsv` filename for the Maximum *A Posteriori* (MAP) reconstruction of presence/absence states at the tips. |
| `-r` | Reconstruct Root | No | Output `.tsv` filename for the ancestral state of gene presence/absence at the root of the phylogeny. |
| `-a` | Ancestral Reconstruction | No | Output `.tsv` filename for the MAP inferred ancestral states using internally assigned node IDs. **Requires `-n` to be specified to interpret the IDs.** |
| `-i` | Inferred Completeness | No | Output `.tsv` filename for the model's estimated completeness of each genome. |
| `-n` | Annotated Tree | No | Output annotated tree file containing the internally assigned node IDs (necessary for mapping outputs from `-a`). |

### Example

The following command executes the phylogenetic occupancy model on the Asgard data, reproducing the analysis from our paper:

```bash
python -m phylo_occupancy -t asgard_tree.tree -d eggnog_ortholog_table.tsv -o tip_probs.tsv -m tip_map.tsv -a ancestral_reconstruction.tsv -i inferred_completeness.tsv -n annotated_tree.tree
```

## Citation
If this model is used in a publication, please cite:...
