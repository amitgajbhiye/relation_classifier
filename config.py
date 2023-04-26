# General configuration
# ---------------------
logging_folder = "logs"

# Configuration for model.py
# --------------------------
model_file_path = "/scratch/c.scmnk4/elexir/resources/learned_models/"
vector_space_dimension = 1024
inner_layer_dimension = 100
batch_size = 5000
epochs = 1000
learning_rate = 0.0025
regularization_constant = 0.1
seed = 1
optimizer_iterations = 10
loss_func = 'MeanCosine'
l2norm_regularization = False
sparsity_constraint = False
reconstruction_error = True
reconstruction_hyper = 0.0
inference_hyper = 0.01
sparsity_constraint_hyper1 = 0.01
sparsity_constraint_hyper2 = 0.00

# Configuration for evaluate.py
# -----------------------------
analogy_dataset = "/scratch/c.scmnk4/elexir/resources/analogy_test_dataset/analogy_test_dataset"
experimental_results = "/scratch/c.scmnk4/elexir/resources/results"
error_analysis_pos_file = "/scratch/c.scmnk4/elexir/resources/results/gpt3_correct_predictions.txt"
error_analysis_neg_file = "/scratch/c.scmnk4/elexir/resources/results/gpt3_incorrect_predictions.txt"
analogy_datasets = ['sat', 'u2', 'u4', 'bats', 'google']
error_analysis = 'on'
weighted_cosine_lambda = 0.95  # None

# Configuration for dictionary.py
# -------------------------------
dictionary_file = "/scratch/c.scmnk4/elexir/resources/dictionary.pl"
relbert_batch_size = 500

# Configuration for knowledge_graph.py
# ------------------------------------
genericsKB = "/scratch/c.scmnk4/elexir/resources/GenericsKB-Best.tsv"
knowledge_graph_file = "/scratch/c.scmnk4/elexir/resources/conceptnet_kb.pl"

# Configuration for dataset.py
# ----------------------------
training_dataset = "/scratch/c.scmnk4/elexir/resources/training_concept_pair_from_semeval_conceptnet_paths_" \
                   "from_numberbatch.pkl"
num_cooccurs = 30
write_to_file_after_size = 50
concept_embedding_model = 'bert-base-uncased'
noisy = True
number_of_concept_pairs_to_consider_to_compute_mean_std = 10000
compute_mean_std_relbert_embeddings = True
noise_weight = 1
std_multiple = 1
store_coling_concept_embeddings_in_memory = True
embedding_name = None
concept_dictionary_tsvfile = "/scratch/c.scmnk4/elexir/resources/dictionary.tsv"
concept_emdeddings = 'COLING'  # 'BERT'
coling_concept_embeddings = "/scratch/c.scmnk4/elexir/resources/learned_models/concept_embeddings/bert_large_mscg_concept_embeddings.pkl"
get_concept_pairs_path_length_3 = True
concept_pairs_maxsize_in_path_length_3 = None
max_number_of_paths_of_len_2 = 20
max_number_of_paths_of_len_3 = 20
get_related_concepts_using_word_embeddings = True
get_related_concepts_using_word_embeddings_topn = 5

# Configuration for model_with_noisy_signal.py
# -------------------------------------------
use_relbert_when_no_intrim_concepts = True

# Configuration for model_dropout.py
# -------------------------------------------
dropout_rate = 0.1

# Configuration for model_dropout_steven.py
# -----------------------------------------
inner_reconstruction_before_dropout = False
activations_debug_folder = '/scratch/c.scmnk4/elexir/resources/learned_models/'

# Configuration for VonMisesFisherMix.py
# --------------------------------------
mixture_model_file = '/scratch/c.scmnk4/elexir/resources/learned_models/von_mises_fisher_mixture_model.pkl'
n_clusters = 3
t_sne_scatter_plot_size_for_each_class = 12800

# Configuration for compressor_relbert_vector.py
# --------------------------------------
compressed_dim = 64
compressor_training_batch = 5500
compressor_number_of_epochs = 10

# Configuration for importance_classifier.py
# ------------------------------------------
classifier_model_path = 'importance_classifier_inverse.pkl'
# '/scratch/c.scmnk4/elexir/resources/learned_models/importance_classifier_inverse.pkl'
# 'log_reg_model.pkl'
# '/scratch/c.scmnk4/elexir/resources/learned_models/classifier_model.pkl'
# classifier_model_path = '/scratch/c.scmnk4/elexir/resources/learned_models/classifier_model2.pkl'
