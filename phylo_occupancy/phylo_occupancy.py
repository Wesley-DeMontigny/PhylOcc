import jax
import jax.numpy as jnp
from jax import lax
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from ete3 import Tree
import pandas as pd

class PhyloOccupancy(dist.Distribution):
    """
    The main class for the phylogenetic occupancy model. Takes in parameters that allow it to perform Felsenstein's pruning algorithm (Variable Elimination)
    with uncertain emission of binary states at the tips.
    
    Args:
        post_order: A post order traversal object. These can be constructed using the construct_post_order method.
        branch_lengths: A jnp float32 array of branch lengths for the tree. These can be provided by the construct_post_order method or as a numpyro parameter.
        completeness: A jnp float32 array of genome completeness values. These can be provided by the user through something like CheckM scores or as a numpyro parameter.
        num_genomes: The number of genomes in the tree.
        num_genes: The number of genes/orthogroups/gene-families in the analysis.
        complete_genomes: A jnp int32 array of the indices of genomes that are thought to be complete (or should be treated as complete by the model).
        core_selection: A jnp int32 array of the indices of genes that are thought to be present in every genome in the dataset.
        num_rates: The number of rate categories in the mixture model (default is 10).
        rate_mean: The mean of the log Normal distribution the rate categories are from (default is 0.0).
        rate_var: The variance of the log Normal distribution the rate categories are from (default is 1.0).
        root_0: The probability of any given gene being absent at the last common ancestor of the lineage (default is 0.5). 
    """
    def __init__(self, post_order : jnp.array, branch_lengths : jnp.array, completeness : jnp.array, num_genomes : int, num_genes : int, 
                 complete_genomes : jnp.array = jnp.empty(0, dtype=jnp.int32), core_selection : jnp.array = jnp.empty(0, dtype=jnp.int32), 
                 num_rates : int = 10, rate_mean : float = 0.0, rate_var : float = 1.0, root_0 : float = 0.5):
        self.post_order = post_order
        self.branch_lengths = branch_lengths
        self.taxa = num_genomes
        self.sites = num_genes
        self.completeness = completeness
        self.complete_genomes = complete_genomes
        self.core_selection = core_selection

        self.num_rates = num_rates
        self.rates = lax.stop_gradient(dist.LogNormal(rate_mean, rate_var).icdf(jnp.arange(1, self.num_rates * 2, 2) / (self.num_rates * 2)))
        self.root_distribution = jnp.tile(jnp.array([root_0, 1.0 - root_0]).T, self.num_rates)

        super().__init__(event_shape = (self.sites, self.taxa), batch_shape = ())

    @property
    def support(self):
        return constraints.boolean

    def _transition_prob(self, branch_length):
        stationary_0 = 0.5
        stationary_1 = 1 - stationary_0
        exp_branches = jnp.exp(-branch_length * self.rates)

        t_probs = jax.vmap(lambda exp_branch:
                          jnp.array([[stationary_0 + stationary_1 * exp_branch, stationary_1 - stationary_1 * exp_branch],
                                     [stationary_0 - stationary_0 * exp_branch, stationary_1 + stationary_0 * exp_branch]]))(exp_branches)

        return jax.scipy.linalg.block_diag(*t_probs)

    def _construct_emission_matrix(self, data):
        dataT = data.T

        completeness_full = self.completeness.at[self.complete_genomes].set(1.0)

        zero_case = jnp.stack([jnp.ones(self.taxa), 1.0 - completeness_full], axis=-1)
        zero_case = jnp.expand_dims(zero_case, 1)
        zero_case = jnp.broadcast_to(zero_case, (self.taxa, self.sites, 2))

        one_case = jnp.stack([jnp.zeros(self.taxa), completeness_full], axis=-1)
        one_case = jnp.expand_dims(one_case, 1)
        one_case = jnp.broadcast_to(one_case, (self.taxa, self.sites, 2))

        zero_core_case = jnp.stack([jnp.zeros(self.taxa), 1.0 - self.completeness], axis=-1)
        zero_core_case = jnp.expand_dims(zero_core_case, 1)
        zero_core_case = jnp.broadcast_to(zero_core_case, (self.taxa, self.sites, 2))

        one_core_case = jnp.stack([jnp.zeros(self.taxa), self.completeness], axis=-1)
        one_core_case = jnp.expand_dims(one_core_case, 1)
        one_core_case = jnp.broadcast_to(one_core_case, (self.taxa, self.sites, 2))

        mask = (dataT == 1)[..., None]

        value = jnp.where(mask, one_case, zero_case)

        core_mask = jnp.isin(jnp.arange(self.sites), self.core_selection)
        core_mask_bc = core_mask[None, :, None]
        core_mask_bc = jnp.broadcast_to(core_mask_bc, (self.taxa, self.sites, 2))
        core_value = jnp.where(mask, one_core_case, zero_core_case)

        value = jnp.where(core_mask_bc, core_value, value)
        value = jnp.concatenate([value] * self.num_rates, axis=-1)

        return value

    def _calc_probs(self, value):
        likelihoods = jnp.zeros(((2*self.taxa)-1, self.sites, 2 * self.num_rates))
        likelihoods = likelihoods.at[:value.shape[0], :, :].set(value)

        def post_order_traversal(carry, post_order_step):
            likelihoods, log_scale_adjustment = carry
            parent, left_child, right_child = post_order_step

            left_likelihood = likelihoods[left_child]
            right_likelihood = likelihoods[right_child]

            left_transition_matrix = self._transition_prob(self.branch_lengths[left_child])
            right_transition_matrix = self._transition_prob(self.branch_lengths[right_child])

            left_likelihood = left_likelihood @ left_transition_matrix
            right_likelihood = right_likelihood @ right_transition_matrix

            parent_likelihood = left_likelihood * right_likelihood

            site_max_likelihood = jnp.max(parent_likelihood, axis=-1) + 1e-12
            parent_likelihood /= site_max_likelihood.reshape([self.sites, 1])

            log_scale_adjustment += jnp.sum(jnp.log(site_max_likelihood))

            likelihoods = likelihoods.at[parent].set(parent_likelihood)

            return (likelihoods, log_scale_adjustment), None

        carry = (likelihoods, 0.0)
        carry, _ = lax.scan(post_order_traversal, carry, self.post_order)

        likelihoods, log_scale_adjustment = carry

        root = self.post_order[-1][0]
        root_likelihood = likelihoods[root]
        total_likelihood = jnp.sum(root_likelihood * self.root_distribution, axis=1)
        log_total_likelihood = jnp.sum(jnp.log(total_likelihood)) + log_scale_adjustment

        return log_total_likelihood, likelihoods

    def log_prob(self, data):
        value = self._construct_emission_matrix(data)
        likelihood, _ = self._calc_probs(value)

        return likelihood

    def sum_product(self, data):
        value = self._construct_emission_matrix(data)
        likelihood, partial_likelihoods = self._calc_probs(value)

        root = self.post_order[-1][0]
        root_likelihood = (partial_likelihoods[root] * self.root_distribution)
        root_sums = jnp.expand_dims(jnp.sum(root_likelihood, axis=1), axis=-1)
        partial_likelihoods = partial_likelihoods.at[root].set(root_likelihood / root_sums)

        def pre_order_traversal(carry, pre_order_step):
            partial_likelihoods = carry
            parent, left_child, right_child = pre_order_step

            left_likelihood = partial_likelihoods[left_child]
            right_likelihood = partial_likelihoods[right_child]
            parent_probs = partial_likelihoods[parent]

            left_transition_matrix = self._transition_prob(self.branch_lengths[left_child])
            right_transition_matrix = self._transition_prob(self.branch_lengths[right_child])
            
            left_child_msg = left_likelihood @ left_transition_matrix
            right_child_msg = right_likelihood @ right_transition_matrix
            
            left_parent_belief = parent_probs / (left_child_msg + 1e-12)
            right_parent_belief = parent_probs / (right_child_msg + 1e-12)
            
            left_trans = left_parent_belief @ left_transition_matrix
            right_trans = right_parent_belief @ right_transition_matrix
            
            unnormalized_left = left_trans * left_likelihood
            unnormalized_right = right_trans * right_likelihood

            left_norm_factor = jnp.expand_dims(jnp.sum(unnormalized_left, axis=1), axis=-1)
            right_norm_factor = jnp.expand_dims(jnp.sum(unnormalized_right, axis=1), axis=-1)

            partial_likelihoods = partial_likelihoods.at[left_child].set(unnormalized_left / left_norm_factor)
            partial_likelihoods = partial_likelihoods.at[right_child].set(unnormalized_right / right_norm_factor)

            return (partial_likelihoods), None

        carry = (partial_likelihoods)
        carry, _ = lax.scan(pre_order_traversal, carry, self.post_order[::-1])

        partial_likelihoods = carry

        return partial_likelihoods

    def max_product(self, data):
        value = self._construct_emission_matrix(data)
        likelihood, partial_likelihoods = self._calc_probs(value)

        reconstructions = jnp.zeros(((2 * self.taxa) - 1, self.sites, 2 * self.num_rates))

        root = self.post_order[-1][0]
        root_likelihood = partial_likelihoods[root] * self.root_distribution
        root_states = jnp.argmax(root_likelihood, axis=1)
        root_one_hot = jax.nn.one_hot(root_states, num_classes=2 * self.num_rates)
        reconstructions = reconstructions.at[root].set(root_one_hot)

        def pre_order_traversal(carry, pre_order_step):
            reconstructions = carry
            parent, left_child, right_child = pre_order_step

            parent_state = reconstructions[parent]

            left_likelihood = partial_likelihoods[left_child]
            right_likelihood = partial_likelihoods[right_child]

            left_transition_matrix = self._transition_prob(self.branch_lengths[left_child])
            right_transition_matrix = self._transition_prob(self.branch_lengths[right_child])

            left_post = (parent_state @ left_transition_matrix) * left_likelihood
            left_states = jnp.argmax(left_post, axis=1)
            left_one_hot = jax.nn.one_hot(left_states, num_classes=2 * self.num_rates)
            reconstructions = reconstructions.at[left_child].set(left_one_hot)

            right_post = (parent_state @ right_transition_matrix) * right_likelihood
            right_states = jnp.argmax(right_post, axis=1)
            right_one_hot = jax.nn.one_hot(right_states, num_classes=2 * self.num_rates)
            reconstructions = reconstructions.at[right_child].set(right_one_hot)

            return (reconstructions), None

        carry = (reconstructions)
        carry, _ = jax.lax.scan(pre_order_traversal, carry, self.post_order[::-1])

        reconstructions = carry

        return reconstructions

    def ancestral_sample(self, data, key):
        value = self._construct_emission_matrix(data)
        likelihood, partial_likelihoods = self._calc_probs(value)

        reconstructions = jnp.zeros(((2 * self.taxa) - 1, self.sites, 2 * self.num_rates))

        root = self.post_order[-1][0]
        root_likelihood = partial_likelihoods[root] * self.root_distribution
        root_sums = jnp.expand_dims(jnp.sum(root_likelihood, axis=1), axis=-1)
        root_probs = root_likelihood / root_sums

        key, subkey = jax.random.split(key)
        unif_root_sample = jax.random.uniform(subkey, shape=(self.sites, 1))
        root_cdf = jnp.cumsum(root_probs, axis=1)
        root_states = jnp.argmax(unif_root_sample < root_cdf, axis=1)
        root_one_hot = jax.nn.one_hot(root_states, num_classes=2 * self.num_rates)
        reconstructions = reconstructions.at[root].set(root_one_hot)

        def pre_order_traversal(carry, pre_order_step):
            reconstructions, key = carry
            key, left_key, right_key = jax.random.split(key, 3)
            parent, left_child, right_child = pre_order_step

            parent_state = reconstructions[parent]

            left_likelihood = partial_likelihoods[left_child]
            right_likelihood = partial_likelihoods[right_child]

            left_transition_matrix = self._transition_prob(self.branch_lengths[left_child])
            right_transition_matrix = self._transition_prob(self.branch_lengths[right_child])

            left_post = (parent_state @ left_transition_matrix) * left_likelihood
            left_post /= jnp.sum(left_post, axis=1, keepdims=True)
            left_cdf = jnp.cumsum(left_post, axis=1)
            left_sample = jax.random.uniform(left_key, shape=(self.sites, 1))
            left_states = jnp.argmax(left_sample < left_cdf, axis=1)
            left_one_hot = jax.nn.one_hot(left_states, num_classes=2 * self.num_rates)
            reconstructions = reconstructions.at[left_child].set(left_one_hot)

            right_post = (parent_state @ right_transition_matrix) * right_likelihood
            right_post /= jnp.sum(right_post, axis=1, keepdims=True)
            right_cdf = jnp.cumsum(right_post, axis=1)
            right_sample = jax.random.uniform(right_key, shape=(self.sites, 1))
            right_states = jnp.argmax(right_sample < right_cdf, axis=1)
            right_one_hot = jax.nn.one_hot(right_states, num_classes=2 * self.num_rates)
            reconstructions = reconstructions.at[right_child].set(right_one_hot)

            return (reconstructions, key), None

        carry = (reconstructions, key)
        carry, _ = jax.lax.scan(pre_order_traversal, carry, self.post_order[::-1])

        reconstructions, _ = carry

        return reconstructions
    
    def sample_node_states(self, key):
        node_states = jnp.zeros(((2 * self.taxa) - 1, self.sites, 2 * self.num_rates))

        root = self.post_order[-1][0]

        key, subkey = jax.random.split(key)
        unif_root_sample = jax.random.uniform(subkey, shape=(self.sites, 1)) * jnp.sum(self.root_distribution)
        root_cdf = jnp.cumsum(self.root_distribution, axis=0)
        root_states = jnp.argmax(unif_root_sample < root_cdf, axis=1)
        root_one_hot = jax.nn.one_hot(root_states, num_classes=2 * self.num_rates)
        node_states = node_states.at[root].set(root_one_hot)

        def pre_order_traversal(carry, pre_order_step):
            node_states, key = carry
            key, left_key, right_key = jax.random.split(key, 3)
            parent, left_child, right_child = pre_order_step

            parent_state = node_states[parent]

            left_transition_matrix = self._transition_prob(self.branch_lengths[left_child])
            right_transition_matrix = self._transition_prob(self.branch_lengths[right_child])

            left_post = (parent_state @ left_transition_matrix)
            left_post /= jnp.sum(left_post, axis=1, keepdims=True)
            left_cdf = jnp.cumsum(left_post, axis=1)
            left_sample = jax.random.uniform(left_key, shape=(self.sites, 1))
            left_states = jnp.argmax(left_sample < left_cdf, axis=1)
            left_one_hot = jax.nn.one_hot(left_states, num_classes=2 * self.num_rates)
            node_states = node_states.at[left_child].set(left_one_hot)

            right_post = (parent_state @ right_transition_matrix)
            right_post /= jnp.sum(right_post, axis=1, keepdims=True)
            right_cdf = jnp.cumsum(right_post, axis=1)
            right_sample = jax.random.uniform(right_key, shape=(self.sites, 1))
            right_states = jnp.argmax(right_sample < right_cdf, axis=1)
            right_one_hot = jax.nn.one_hot(right_states, num_classes=2 * self.num_rates)
            node_states = node_states.at[right_child].set(right_one_hot)

            return (node_states, key), None

        carry = (node_states, key)
        carry, _ = jax.lax.scan(pre_order_traversal, carry, self.post_order[::-1])

        node_states, _ = carry

        return node_states
    
    def sample(self, key):
        node_states = self.sample_node_states(key)
        true_val = jnp.sum(node_states[:self.taxa, :, 1:(2*self.num_rates):2].T, axis=0)
        obs_mask = dist.Binomial(total_count = 1, probs = self.completeness).sample(key, (self.sites,))
        
        return jnp.where(obs_mask == 1, true_val, 0)
        
    def simulate(self, key):
        node_states = self.sample_node_states(key)
        root_val = jnp.sum(node_states[self.post_order[-1][0], :, 1:(2*self.num_rates):2].T, axis=0)
        true_val = jnp.sum(node_states[:self.taxa, :, 1:(2*self.num_rates):2].T, axis=0)
        true_node_states = jnp.sum(node_states[:, :, 1:(2*self.num_rates):2].T, axis=0)
        obs_mask = dist.Binomial(total_count = 1, probs = self.completeness).sample(key, (self.sites,))
        
        return root_val, true_val, jnp.where(obs_mask == 1, true_val, 0), true_node_states

    def marginal_reconstruct_tips(self, data):
        reconstruction = self.sum_product(data)
        return jnp.sum(reconstruction[:data.shape[1], :, 1:(2*self.num_rates):2].T, axis=0)

    def marginal_reconstruct_root(self, data):
        reconstruction = self.sum_product(data)
        return jnp.sum(reconstruction[self.post_order[-1][0], :, 1:(2*self.num_rates):2].T, axis=0)

    def joint_reconstruct_tips(self, data):
        reconstruction = self.max_product(data)
        return jnp.sum(reconstruction[:data.shape[1], :, 1:(2*self.num_rates):2].T, axis=0)
        
    def map_reconstruct(self, data):
        reconstruction = self.max_product(data)
        return jnp.sum(reconstruction[:data.shape[1], :, 1:(2*self.num_rates):2].T, axis=0)

    def joint_sample_tips(self, data, key):
        reconstruction = self.ancestral_sample(data, key)
        return jnp.sum(reconstruction[:data.shape[1], :, 1:(2*self.num_rates):2].T, axis=0)

def model(detection_table=None, num_genomes=None, num_genes=None, post_order=None, complete_genomes=None, core_selection=None):
    branch_lengths = numpyro.param("branch_lengths", jnp.ones([num_genomes*2-2]), constraint=constraints.positive)

    completeness = numpyro.param("completeness", jnp.ones(num_genomes) * 0.5, constraint=constraints.unit_interval)

    phylo_occupancy = numpyro.sample("phylo_occupancy", PhyloOccupancy(post_order, branch_lengths, completeness,
                                                                   num_genomes, num_genes, complete_genomes, core_selection), obs = detection_table)
def construct_post_order(newick, data=None):
    """
    Takes in a newick string of a rooted tree and a corresponding pandas dataframe and produces a labeled ETE3 tree object with node_id metadata, 
    and a post order traversal for the model. Note that for users wanting to perform ancestral state reconsturction at an arbitrary node on the 
    tree, the node_id metadata should be used to select that from ancestral_sample.
    
    Args:
        newick: The newick of the rooted phylogeny.
        data: The pandas dataframe with named columns for the genomes. This can be set to None to not match the column orders.
    """
    tree = Tree(newick, format=1)
    post_order = []
    node_to_id = {}
    internal_index = 0
    
    if data is not None:
        for leaf in tree.iter_leaves():
            node_to_id[leaf] = data.columns.get_loc(leaf.get_leaf_names()[0].replace("'", ""))
    else:
        c = 0
        for leaf in tree.iter_leaves():
            node_to_id[leaf] = c
            c += 1

    num_tips = len(node_to_id)

    def assign_ids_and_traverse(node):
        nonlocal internal_index

        if node not in node_to_id:
            node_to_id[node] = num_tips + internal_index
            internal_index += 1

        children = node.get_children()
        child_ids = []

        for child in children:
            child_id = assign_ids_and_traverse(child)
            child_ids.append(child_id)

        if children:
            post_order.append((node_to_id[node], *child_ids))

        return node_to_id[node]

    assign_ids_and_traverse(tree)
    
    for key in node_to_id:
        key.add_feature("node_id", node_to_id[key])

    return tree, jnp.array(post_order)

def fit_model(data, post_order, complete_genomes=jnp.empty(0, dtype=jnp.int32), 
              core_selection=jnp.empty(0, dtype=jnp.int32), 
              num_iterations=1000, learning_rate=1e-2, random_key=jax.random.PRNGKey(0)):
    """
    Returns an SVIRunResult object corresponding to the maximum likelihood estimation of the PhyloOccupancy model.

    Args:
        data: A jnp float32 array with shape (num_genes, num_genomes) containing presence/absence data (1/0).
        post_order: A post-order traversal array of a rooted phylogenetic tree corresponding to the dataset.
        complete_genomes: A jnp int32 array containing the indices of the genomes (if any) that are considered complete.
        core_selection: A jnp int32 array containing the indices of the gene families (if any) that are thought to be present in all genomes.
        num_iterations: The number of iterations to run the Adam optimizer. In our experience 1000 iterations is a reasonable default, but this may vary on your data.
        learning_rate: The learning rate for the Adam optimizer.
        random_key: The random key used by the optimizer.
    """

    num_genes, num_genomes = data.shape
    
    loss = numpyro.infer.Trace_ELBO(num_particles=1)
    svi = numpyro.infer.SVI(model, guide=lambda *args, **kwargs: None, optim=numpyro.optim.Adam(learning_rate), loss=loss)
    fit = svi.run(random_key, num_iterations, detection_table=data, num_genomes=num_genomes, num_genes=num_genes, 
                  post_order=post_order, complete_genomes=complete_genomes, core_selection=core_selection, progress_bar=True)
    return fit
