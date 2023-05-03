import sys
import numpy as np
from packaging import version
from relbert import RelBERT

import config
from sklearn.linear_model import LogisticRegression

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


file = "sorted_counts_numberbatch_con_similarsim_thresh_50_count_thresh_60_12.txt"
# file = "temp.txt"

con_sim_list = read_data(file_path=file)

con_sim_list = [t.split("\t") for t in con_sim_list]

if __name__ == "__main__":
    classifier = Classifier()
    classifier.load_model()

    model = RelBERT(
        "relbert/relbert-roberta-large",
    )
    embeddings = model.get_embedding(con_sim_list, batch_size=512)

    importance = classifier.importance(embeddings)

    # print(type(importance))
    # print(importance, flush=True)

    with open("numberbatch_relation_probs.txt", "w") as out_file:
        for (con1, con2), score in zip(con_sim_list, importance):
            print((f"{con1} &&& {con2} &&& {round(score, 4)}"), flush=True)
            print(flush=True)
            out_file.write(f"{con1}|{con2}|{round(score, 4)}\n")
