import sys
import math
import numpy as np
from packaging import version
from relbert import RelBERT

import logging
import os

import config
from sklearn.linear_model import LogisticRegression
from os import listdir


def initialization():
    log_file = os.path.join(
        "logs", os.path.splitext(os.path.basename(__file__))[0] + ".log"
    )

    logging.basicConfig(
        format="%(levelname)s %(asctime)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_file,
        filemode="w",
        level=logging.INFO,
    )


if version.parse(
    str(sys.version_info[0]) + "." + str(sys.version_info[1])
) < version.parse("3.9"):
    import pickle5 as pickle
else:
    import pickle


class Classifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42, verbose=1)

    def load_model(self):
        with open(config.classifier_model_path, "rb") as f:
            self.model = pickle.load(f)

    def importance1(self, data):
        data = np.vstack(data)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        probabilities = self.model.predict_proba(data)
        importance = []
        for [_, b] in probabilities:
            importance.append(b)
        return importance

    def importance(self, data):
        if type(data[0]) is list:
            importance = self.importance1(data)
        else:
            importance = self.importance1([data])[0]
        return importance


def read_data(file_path):
    with open(file_path, "r") as in_file:
        lines = in_file.read().splitlines()

    return lines


if __name__ == "__main__":
    initialization()

    logging.info("Running RelBERT Based Importance Classifier...")

    # Reading Commandline arguments
    print(f"Input Arguments : {sys.argv}", flush=True)
    _, inp_dir = sys.argv

    inp_files = [
        os.path.join(inp_dir, fname)
        for fname in listdir(inp_dir)
        if fname.startswith("for_relbert_scoring")
    ]
    out_files = [
        filename.replace("for_relbert_scoring", "relbert_scored")
        for filename in inp_files
    ]

    print(f"inp_files : {inp_files}")
    print(f"out_files : {out_files}")

    for inp_file, out_file in zip(inp_files, out_files):
        print(f"Processing", flush=True)
        print(f"input_fle: {inp_file}", flush=True)
        print(f"output_file: {out_file}", flush=True)

        # Input Concepts Similar Data
        con_sim_list = read_data(file_path=inp_file)
        con_sim_list = [t.split("\t") for t in con_sim_list]

        print(f"record_num_inp_file : {len(con_sim_list)}")
        print(flush=True)

        classifier = Classifier()
        classifier.load_model()

        model = RelBERT(
            "relbert/relbert-roberta-large",
        )

        batch_size = 10000
        all_embeddings = []
        batch_counter = 1

        total_batches = math.ceil(len(con_sim_list) / batch_size)

        with open(out_file, "w") as out_file:
            for i in range(0, len(con_sim_list), batch_size):
                con_sim_batch = con_sim_list[i : i + batch_size]
                embeddings = model.get_embedding(con_sim_batch, batch_size=2048)

                print(
                    f"processing_batch : {batch_counter} / {total_batches}", flush=True
                )

                importance = classifier.importance(embeddings)

                print(f"len_con_sim_batch, {len(con_sim_batch)}", flush=True)
                print(f"len_embeddings, {len(embeddings)}", flush=True)
                print(f"len_importance, {len(importance)}", flush=True)

                assert (
                    len(con_sim_batch) == len(embeddings) == len(importance)
                ), f"len con_sim_list: {len(con_sim_list)}, len embeddings: {len(embeddings)}, len importance: {len(importance)}, not equal"

                for (con1, con2), score in zip(con_sim_batch, importance):
                    print((f"{con1} &&& {con2} &&& {round(score, 4)}\n"), flush=True)
                    out_file.write(f"{con1}\t{con2}\t{round(score, 4)}\n")

                print(
                    f"finished_processing_batch : {batch_counter}\n",
                    flush=True,
                )
                batch_counter += 1


# ++++++++++++++++++++++++++++++
# Word2vec
# file = "datasets/rel_inp_word2vec_ueft_label_similar_0.5thresh_count_10thresh.txt"
# out_file = "output_files/w2v_relation_probs.txt"


# Numberbatch
# file = "datasets/rel_inp_numberbatch_ueft_label_similar_0.5thresh_count_20thresh.txt"
# out_file = "output_files/numberbatch_relation_probs.txt"


# Fasttext
# file = "datasets/rel_inp_fasttext_ueft_label_similar_0.5thresh_count_100thresh.txt"
# out_file = "output_files/fasttext_relation_probs.txt"


# if __name__ == "__main__":
#     initialization()

#     logging.info("Running RelBERT Based Importance Classifier...")

#     # Reading Commandline arguments
#     print(f"Input Arguments : {sys.argv}", flush=True)
#     _, inp_file, out_file = sys.argv

#     print(flush=True)
#     print(f"input_fle: {inp_file}", flush=True)
#     print(f"output_file: {out_file}", flush=True)

#     # Input Concepts Similar Data
#     con_sim_list = read_data(file_path=inp_file)
#     con_sim_list = [t.split("\t") for t in con_sim_list]

#     print(f"record_num_inp_file : {len(con_sim_list)}")
#     print(flush=True)

#     classifier = Classifier()
#     classifier.load_model()

#     model = RelBERT(
#         "relbert/relbert-roberta-large",
#     )

#     batch_size = 10000
#     all_embeddings = []
#     batch_counter = 1

#     total_batches = math.ceil(len(con_sim_list) / batch_size)

#     with open(out_file, "w") as out_file:
#         for i in range(0, len(con_sim_list), batch_size):
#             con_sim_batch = con_sim_list[i : i + batch_size]
#             embeddings = model.get_embedding(con_sim_batch, batch_size=2048)

#             print(f"processing_batch : {batch_counter} / {total_batches}", flush=True)

#             importance = classifier.importance(embeddings)

#             print(f"len_con_sim_batch, {len(con_sim_batch)}", flush=True)
#             print(f"len_embeddings, {len(embeddings)}", flush=True)
#             print(f"len_importance, {len(importance)}", flush=True)

#             assert (
#                 len(con_sim_batch) == len(embeddings) == len(importance)
#             ), f"len con_sim_list: {len(con_sim_list)}, len embeddings: {len(embeddings)}, len importance: {len(importance)}, not equal"

#             for (con1, con2), score in zip(con_sim_batch, importance):
#                 print((f"{con1} &&& {con2} &&& {round(score, 4)}\n"), flush=True)
#                 out_file.write(f"{con1}|{con2}|{round(score, 4)}\n")

#             print(
#                 f"finished_processing_batch : {batch_counter}\n",
#                 flush=True,
#             )
#             batch_counter += 1
