from ete3 import Tree
import argparse
from scipy.stats import beta
import pandas as pd
import numpyro
import jax
import jax.numpy as jnp
import phylo_occupancy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Phylogenetic Occupancy Model")

    parser.add_argument(
        "-t", "--tree", required=True,
        help="Input rooted phylogeny in Newick format."
    )
    parser.add_argument(
        "-d", "--data", required=True,
        help="Input TSV file of presence/absence data. First column = ortholog names; first row = genome names."
    )
    parser.add_argument(
        "-c", "--core_genes", required=False,
        help="Comma-separated list of gene indices expected to be present in all genomes (no spaces)."
    )
    parser.add_argument(
        "-g", "--complete_genomes", required=False,
        help="Comma-separated list of genome indices expected to be complete (no spaces)."
    )
    parser.add_argument(
        "-o", "--tip_probabilities", required=False, default="tip_probabilities.tsv",
        help="Output file for marginal presence probabilities at the tips (default: tip_probabilities.tsv)."
    )
    parser.add_argument(
        "-m", "--tip_map", required=False, default="tip_map_states.tsv",
        help="Output file for maximum a posteriori estimate of the states at the tips (default: tip_map_states.tsv)."
    )
    parser.add_argument(
        "-r", "--reconstruct_root", required=False,
        help="Output file for root ancestral marginal probabilities."
    )
    parser.add_argument(
        "-a", "--ancestral_reconstruction", required=False,
        help="Output file for marginal presence probabilities at all nodes in the phylogeny."
    )
    parser.add_argument(
        "-i", "--inferred_completeness", required=False,
        help="Output file for inferred completeness values of the input genomes."
    )
    parser.add_argument(
        "-n", "--annotated_tree", required=False,
        help="Output annotated Newick tree including node IDs used in the model. Useful to identify nodes of interest in ancestral state reconstructions."
    )
    args = parser.parse_args()
    
    print("Loading the data...")
    
    tree_input = Tree(args.tree, format=1)
    input_df = pd.read_csv(args.data, index_col=0, sep="\t")
    
    tree_obj, post_order = phylo_occupancy.construct_post_order(tree_input.write(format=1), input_df)
    leaf_names = tree_obj.get_leaf_names()
    leaf_names = [x.strip("'\"") for x in leaf_names]
    for n in leaf_names:
        if n not in input_df.columns:
            raise Exception(f"{n} is not present in the provided data table!")
    
    complete_genomes = jnp.empty(0, dtype=jnp.int32)
    if args.complete_genomes is not None:
        complete_genomes = jnp.array(str(args.complete_genomes).split(","), dtype=jnp.int32)

    core_genes = jnp.empty(0, dtype=jnp.int32)
    if args.core_genes is not None:
        core_genes = jnp.array(str(args.core_genes).split(","), dtype=jnp.int32)

    data = jnp.array(input_df.values, dtype=jnp.float32)
    num_genes, num_genomes = data.shape
    
    print("Starting phylogenetic occupancy model...")
    
    fit = phylo_occupancy.fit_model(data, post_order, complete_genomes, core_genes)

    phylo_occupancy_dist = phylo_occupancy.PhyloOccupancy(post_order, fit.params["branch_lengths"], fit.params["completeness"], num_genomes, num_genes, complete_genomes, core_genes)
    
    a1, b1, loc1, scale1 = beta.fit(fit.params["completeness"], floc=0, fscale=1)
    print(f"Inferred completeness distribution is approximately a Beta({a1},{b1})")
    
    print("Reconstructing marginal tip probabilities...")
    
    marginal_probs = pd.DataFrame(phylo_occupancy_dist.marginal_reconstruct_tips(data), columns=input_df.columns)
    marginal_probs = marginal_probs.set_index(input_df.index)
    marginal_probs.to_csv(args.tip_probabilities, sep="\t")
    
    print("Reconstructing MAP estimate...")
    
    joint_map = pd.DataFrame(phylo_occupancy_dist.joint_reconstruct_tips(data), columns=input_df.columns)
    joint_map = joint_map.set_index(input_df.index)
    joint_map.to_csv(args.tip_map, sep="\t")
    
    if args.reconstruct_root is not None:
        print("Reconstructing the tree root...")
        root_reconstruction = pd.DataFrame(phylo_occupancy_dist.marginal_reconstruct_root(data), columns=["root"])
        root_reconstruction = root_reconstruction.set_index(input_df.index)
        root_reconstruction.to_csv(args.reconstruct_root, sep="\t")
    
    if args.ancestral_reconstruction is not None:
        print("Performing whole (MAP) ancestral state reconstruction...")
        reconstruction = phylo_occupancy_dist.max_product(data)
        ancestral_reconstruction = pd.DataFrame(jnp.sum(reconstruction[:, :, 1:20:2].T, axis=0))
        ancestral_reconstruction = ancestral_reconstruction.set_index(input_df.index)
        ancestral_reconstruction.to_csv(args.ancestral_reconstruction, sep="\t")
    
    if args.inferred_completeness is not None:
        inferred_completeness = pd.DataFrame(fit.params["completeness"])
        inferred_completeness = inferred_completeness.set_index(input_df.columns)
        inferred_completeness.to_csv(args.inferred_completeness, sep="\t")
    
    if args.annotated_tree is not None:
        tree_obj.write(outfile=args.annotated_tree, format=1, features=["node_id"])