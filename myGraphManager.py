import traceback
import datetime
import dateutil.parser
import pandas as pd
import csv
import json
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import networkx as nx
import math
import community as community_louvain
from myPreProcessor import MyPreProcessor
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn import preprocessing
from pywaffle import Waffle
import seaborn as sns


def directory_manipulation(path):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    os.mkdir(path)


def decision_function_nodes(value):
    # return 1 - (1 / (1 + 10 * pow(10, -2 * value)))
    return 1 / pow(10, value)


class MyGraphManager:

    def __init__(self, pure_init_directory, pure_init_file, year, top_n_tf_idf_limit, save_results):
        self.directory_path = Path(pure_init_directory + "/" + str(year))
        self.jsons_directory_path = Path(self.directory_path.__str__() + "/jsons")
        self.profiles_directory_path = Path(self.directory_path.__str__() + "/profiles")
        self.results_directory_path = Path(self.directory_path.__str__() + "/results")
        self.pure_init_directory = pure_init_directory
        self.pure_init_file = pure_init_file
        self.year = year
        self.tf_idf_vectorizer = TfidfVectorizer(max_features=top_n_tf_idf_limit)
        self.count_vectorizer = CountVectorizer(analyzer="char_wb")
        self.pre_processor = MyPreProcessor()
        self.csv_newline = ""
        self.csv_encoding = "utf-8"
        self.save_results = save_results
        self.round_decimals = 8
        self.edge_anomaly_space = 0.05
        self.edge_anomaly_distance = 0.50 - self.edge_anomaly_space
        self.community_anomaly_space = 2 * self.edge_anomaly_space
        self.community_anomaly_distance = 0.50 - self.community_anomaly_space

    def data_manipulation(self, given_text, join_string=None):
        text_list = self.pre_processor.data_pre_processing_nltk(given_text)
        if join_string is None:
            return text_list
        list_to_string = join_string.join([str(element) for element in text_list])
        return list_to_string

    def data_manipulation_extended(self, given_text, join_string=None):
        text_list = self.pre_processor.data_pre_processing_nltk_extended(given_text)
        if join_string is None:
            return text_list
        list_to_string = join_string.join([str(element) for element in text_list])
        return list_to_string

    def jaccard_similarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def get_jaccard_similarity_between_two_strings(self, strings_array):
        vector0 = strings_array[0].split(" ")
        vector1 = strings_array[1].split(" ")
        similarity_score = self.jaccard_similarity(vector0, vector1)
        # vectors = self.count_vectorizer.fit_transform(strings_array).toarray()
        # similarity_score = jaccard_score(vector0, vector1, average="samples")
        return round(similarity_score, self.round_decimals)

    def csv_write(self, csv_file, csv_columns, dict_data):
        with open(csv_file, "w", newline=self.csv_newline, encoding=self.csv_encoding) as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writeheader()
            for item in dict_data:
                writer.writerow(dict_data[item])

    def result_write(self,
                     X,
                     Y,
                     title,
                     x_label,
                     y_label,
                     result_type="D"):
        f = None
        if self.save_results:
            f = plt.figure()

        if result_type == "D":
            data = pd.DataFrame(Y, columns=X, index=["Positive", "Negative"])
            ax = sns.heatmap(data, annot=True)
            # plt.show()

        if result_type == "S":
            plt.scatter(X, Y)
            x = np.array([0, 1])
            y = np.array([0, 1])
            plt.plot(x, y, color="r")

            cx1 = np.array([0, 1 - self.community_anomaly_distance])
            cy1 = np.array([self.community_anomaly_distance, 1])
            plt.plot(cx1, cy1, linestyle="dashed", color="m")
            cx2 = np.array([self.community_anomaly_distance, 1])
            cy2 = np.array([0, 1 - self.community_anomaly_distance])
            plt.plot(cx2, cy2, linestyle="dashed", color="m")

            ex1 = np.array([0, 1 - self.edge_anomaly_distance])
            ey1 = np.array([self.edge_anomaly_distance, 1])
            plt.plot(ex1, ey1, linestyle="dotted", color="c")
            ex2 = np.array([self.edge_anomaly_distance, 1])
            ey2 = np.array([0, 1 - self.edge_anomaly_distance])
            plt.plot(ex2, ey2, linestyle="dotted", color="c")

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if self.save_results:
            f.savefig(self.results_directory_path.__str__() + "/" + title + ".pdf", bbox_inches="tight")
        else:
            plt.show()

    def manipulate_data(self):
        init_time = None
        end_time = None

        hash_map_nodes = {}
        hash_map_edges = {}

        try:
            directory_manipulation(self.jsons_directory_path)

            df = pd.read_csv(self.pure_init_directory + "/" + self.pure_init_file)

            df = df[(pd.notna(df["publish_time"]))]
            df = df[["publish_time", "pdf_json_files", "pmc_json_files"]]
            df["publish_time"] = df["publish_time"].apply(lambda x: dateutil.parser.parse(str(x)).year)
            df_unfiltered_index = str(len(df.index))

            df_filtered = df[(pd.notna(df["pdf_json_files"])) | (pd.notna(df["pmc_json_files"]))]
            df_filtered_index = str(len(df_filtered.index))

            df_filtered = df_filtered.groupby("publish_time")

            grouped_df = df_filtered.get_group(self.year)
            df_grouped_index = str(len(grouped_df.index))

            paper_iteration_index = 0

            print("========================\nSelection on Graph data Starts")
            init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            for paper_index, row in grouped_df.iterrows():
                paper_iteration_index += 1

                pdf_json_file = row["pdf_json_files"]
                pmc_json_file = row["pmc_json_files"]
                print("------------------------\nPaper iteration Index: " + str(paper_iteration_index) + "")
                print("PDF json File: " + str(pdf_json_file))
                print("PCM json File: " + str(pmc_json_file))
                print("Paper Index: " + str(paper_index))

                if pd.isna(pdf_json_file):
                    pdf_json_file = pmc_json_file

                if pd.isna(pmc_json_file):
                    pmc_json_file = pdf_json_file

                pdf_json_file = pdf_json_file.split(";")[0]
                pmc_json_file = pmc_json_file.split(";")[0]

                with open(self.pure_init_directory + "/" + pdf_json_file) as pdf_file, open(self.pure_init_directory + "/" + pmc_json_file) as pmc_file:
                    data_pdf = json.load(pdf_file)
                    data_pcm = json.load(pmc_file)

                    authors = data_pdf["metadata"]["authors"]
                    if len(authors) == 0:
                        authors = data_pcm["metadata"]["authors"]

                    paper_text_clear = ""
                    if len(authors) != 0:

                        paper_text = data_pdf["metadata"]["title"]
                        if len(paper_text) == 0:
                            paper_text = data_pdf["metadata"]["title"]

                        paragraphs = data_pdf["body_text"]
                        paragraphs_pdf = ""
                        for item in paragraphs:
                            paragraphs_pdf = paragraphs_pdf + item.get("text")

                        paragraphs = data_pcm["body_text"]
                        paragraphs_pcm = ""
                        for item in paragraphs:
                            paragraphs_pcm = paragraphs_pcm + item.get("text")

                        paragraphs = paragraphs_pdf
                        if len(paragraphs_pcm) > len(paragraphs_pdf):
                            paragraphs = paragraphs_pcm

                        paper_text = paper_text + paragraphs

                        paper_text_clear = self.data_manipulation_extended(paper_text, " ")

                    authors_as_list = []

                    print("Status: " + "Paper's Clear Body Text Prepared")

                    if paper_text_clear != "":
                        set_of_jsons = {json.dumps(d, sort_keys=True) for d in authors}
                        authors = [json.loads(t) for t in set_of_jsons]
                        for author in authors:
                            if len(author["middle"]) == 0:
                                author_text_clear = author["first"] + author["last"]
                            else:
                                middle = "".join([str(element) for element in author["middle"]])
                                author_text_clear = author["first"] + middle + author["last"]
                            author_text_clear = self.data_manipulation(author_text_clear, " ").strip()

                            if author_text_clear != "" and (author_text_clear not in authors_as_list):
                                authors_as_list.append(author_text_clear)
                                key = hash(author_text_clear)

                                if key not in hash_map_nodes:
                                    hash_map_nodes[key] = {
                                        "id": str(key),
                                        "name": author_text_clear,
                                        "length": len(paper_text_clear),
                                        "papers_array": [{"paper_id_pdf": data_pdf["paper_id"], "paper_id_pcm": data_pcm["paper_id"], "text": paper_text_clear}]
                                    }
                                    with open(self.jsons_directory_path.__str__() + "/" + str(key) + ".txt", "w") as json_file:
                                        json.dump(hash_map_nodes[key]["papers_array"], json_file)
                                else:
                                    hash_map_nodes[key]["length"] += len(paper_text_clear)
                                    hash_map_nodes[key]["papers_array"].append({"paper_id_pdf": data_pdf["paper_id"], "paper_id_pcm": data_pcm["paper_id"], "text": paper_text_clear})
                                    with open(self.jsons_directory_path.__str__() + "/" + str(key) + ".txt", "w") as json_file:
                                        json.dump(hash_map_nodes[key]["papers_array"], json_file)

                            print("Status: " + "Authors(Nodes) Paper's Prepared")

                        if len(authors_as_list) == 1:

                            print("Status: " + "Author-Pairs Number is 1")

                        else:

                            authors_as_list.sort()

                            authors_r_subset = list(combinations(authors_as_list, 2))

                            print("Status: " + "Author-Pairs Number is " + str(len(authors_r_subset)))

                            for authors_pair in authors_r_subset:
                                hashed_author_source = hash(authors_pair[0])
                                hashed_author_target = hash(authors_pair[1])
                                if hashed_author_source != hashed_author_target:
                                    key = hash(str(hashed_author_source) + str(hashed_author_target))
                                    if key not in hash_map_edges:
                                        hash_map_edges[key] = {
                                            "id": str(key),
                                            "source": str(hashed_author_source),
                                            "target": str(hashed_author_target),
                                            "weight": 1,
                                            "length": len(paper_text_clear),
                                            "undirected": "Undirected"
                                        }
                                    else:
                                        hash_map_edges[key]["weight"] += 1
                                        hash_map_edges[key]["length"] += len(paper_text_clear)

                        print("Status: " + "Author-Pairs(Edges) Prepared")
            end_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            print("Selection on Graph data Prepared\n========================")

            df_hash_map_nodes = pd.DataFrame.from_dict(hash_map_nodes, orient="index").T
            df_hash_map_nodes = df_hash_map_nodes.drop("papers_array")
            hash_map_nodes = df_hash_map_nodes.to_dict()

            self.csv_write(self.directory_path.__str__() + "/gephiNodes.csv", ["id", "name", "length"], hash_map_nodes)
            self.csv_write(self.directory_path.__str__() + "/gephiEdges.csv", ["id", "source", "target", "weight", "length", "undirected"], hash_map_edges)
            print("========================\nmanipulate_data: Run time details")
            print("Unfiltered Index: " + df_unfiltered_index)
            print("Filtered Index: " + df_filtered_index)
            print("Grouped Index: " + df_grouped_index)
            print("------------------------")
            print(init_time)
            print(end_time)
            print("manipulate_data: Run time details\n========================")

        except Exception as error:
            print("===================================================================================================")
            print("Error on_main: %s" % str(error))
            print("===================================================================================================")
            traceback.print_exc()
            print("===================================================================================================")
            print(init_time)
            print(end_time)

    def manipulate_profiles(self):
        init_time = None
        end_time = None

        try:
            directory_manipulation(self.profiles_directory_path)

            print("========================\nProfile Manipulation on Graph nodes Starts")
            init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            files = os.listdir(self.jsons_directory_path.__str__())
            for key in files:
                with open(self.jsons_directory_path.__str__() + "/" + str(key)) as json_file:
                    json_array = json.load(json_file)
                    if json_array[0]["text"] == "":
                        profile_tf_idf = ""
                    else:
                        papers_array = []
                        for item in json_array:
                            papers_array.append(item["text"])
                        self.tf_idf_vectorizer.fit_transform(papers_array)
                        feature_array = np.array(self.tf_idf_vectorizer.get_feature_names())
                        profile_tf_idf = " ".join(feature_array)
                    with open(self.profiles_directory_path.__str__() + "/" + str(key), "w") as profile_file:
                        json.dump({"text": profile_tf_idf}, profile_file)
            end_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            print("Profile Manipulation on Graph nodes Prepared\n========================")

            print("========================\nmanipulate_profiles: Run time details")
            print(init_time)
            print(end_time)
            print("manipulate_profiles: Run time details\n========================")

        except Exception as error:
            print("===================================================================================================")
            print("Error on_main: %s" % str(error))
            print("===================================================================================================")
            traceback.print_exc()
            print("===================================================================================================")
            print(init_time)
            print(end_time)

    def df_reader(self, row, hash_map_nodes, column_type):
        length = hash_map_nodes[row[column_type]]["length"]
        path = self.profiles_directory_path.__str__() + "/" + str(row[column_type]) + ".txt"
        with open(path) as profile_file:
            json_profile = json.load(profile_file)
            json_profile_text = json_profile["text"]
        return pd.Series([json_profile_text, length], index=["text_" + column_type, "length_" + column_type])

    def df_weighter(self, row):
        total = row["length_source"] + row["length_target"] - row["length"] + 1
        return round(row["length"]/total, self.round_decimals)

    def df_jaccarder(self, row):
        return self.get_jaccard_similarity_between_two_strings([row["text_source"], row["text_target"]])

    def df_anomaler(self, row):
        jaccard_anomaly_score = row["jaccard"] - row["weight"]
        return pd.Series([row["id"], row["source"], row["target"], jaccard_anomaly_score, row["jaccard"], row["weight"]
                          ], index=["id", "source", "target", "jaccard_anomaly_score", "jaccard_similarity_score", "weight"])

    def manipulate_results(self, sub_graph_nodes=[],
                           get_anomaly_edges=False,
                           get_anomaly_nodes=False,
                           get_anomaly_communities=False):
        init_time = None
        end_time = None

        try:
            directory_manipulation(self.results_directory_path)

            init_time = None
            end_time = None

            print("========================\nScores Calculation Starts")
            init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            print(init_time)

            df = pd.read_csv(self.directory_path.__str__() + "/gephiNodes.csv")
            hash_map_nodes = df.set_index("id").T.to_dict()
            df = pd.read_csv(self.directory_path.__str__() + "/gephiEdges.csv")
            df = df.astype("category")
            df = df.iloc[56000:87000]
            df_source = df.apply(lambda x: self.df_reader(x, hash_map_nodes, "source"), axis=1)
            df_source = df_source.astype("category")
            df_target = df.apply(lambda x: self.df_reader(x, hash_map_nodes, "target"), axis=1)
            hash_map_nodes = None
            df_target = df_target.astype("category")
            df = pd.concat([df["id"], df["source"], df["target"], df_source, df_target, df["length"]], axis=1, join="inner")
            df = df.astype("category")
            df_source = None
            df_target = None
            df_weight = df.apply(self.df_weighter, axis=1)
            df_weight = df_weight.astype("category")
            df_jaccard = df.apply(self.df_jaccarder, axis=1)
            df_jaccard = df_jaccard.astype("category")
            df = pd.concat([df["id"], df["source"], df["target"], df_jaccard, df_weight], axis=1, join="inner")
            df = df.rename(columns={0: "jaccard", 1: "weight"})
            df = df.rename(columns={0: "jaccard"})
            df = df.astype("category")
            df_out_weight = None
            df_jaccard = None
            df = df.apply(self.df_anomaler, axis=1)
            # anomaly_score_column = df["jaccard_anomaly_score"].values.reshape(-1, 1)
            # anomaly_score_column = anomaly_score_column * -1
            # anomaly_score_column = np.vstack([anomaly_score_column, 1])
            # anomaly_score_column = np.vstack([anomaly_score_column, -1])
            # anomaly_score_column_scaled = preprocessing.MinMaxScaler().fit_transform(anomaly_score_column)
            # anomaly_score_column_scaled = np.delete(anomaly_score_column_scaled, len(anomaly_score_column_scaled)-1, 0)
            # anomaly_score_column_scaled = np.delete(anomaly_score_column_scaled, len(anomaly_score_column_scaled)-1, 0)
            # df["jaccard_anomaly_score"] = anomaly_score_column_scaled
            df_eliminated = df[["source", "target", "jaccard_anomaly_score"]]

            init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            print(init_time)
            print("Scores Calculation Ends")

            if get_anomaly_edges:
                print("========================\nEdges Results Starts")
                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)

                del df["source"]
                del df["target"]
                hash_map_jaccard_anomalies = df.to_dict("index")
                self.csv_write(self.directory_path.__str__() + "/results/jacardAnomalies.csv", ["id", "jaccard_anomaly_score", "jaccard_similarity_score", "weight"], hash_map_jaccard_anomalies)

                jaccard_anomaly_count_pos = df["jaccard_anomaly_score"][df["jaccard_anomaly_score"] > self.edge_anomaly_distance].count()
                jaccard_anomaly_count_neg = df["jaccard_anomaly_score"][df["jaccard_anomaly_score"] < -1 * self.edge_anomaly_distance].count()
                jaccard_non_anomaly_count_pos = df["jaccard_anomaly_score"][(df["jaccard_anomaly_score"] <= self.edge_anomaly_distance) & (df["jaccard_anomaly_score"] > 0.0)].count()
                jaccard_non_anomaly_count_neg = df["jaccard_anomaly_score"][(df["jaccard_anomaly_score"] > -1 * self.edge_anomaly_distance) & (df["jaccard_anomaly_score"] <= 0.0)].count()
                distribution_heatmap = [
                    [jaccard_non_anomaly_count_pos, jaccard_anomaly_count_pos],
                    [jaccard_non_anomaly_count_neg, jaccard_anomaly_count_neg]
                ]
                jaccard_anomalies = df_eliminated["jaccard_anomaly_score"].tolist()
                jaccard_similarities = df["jaccard_similarity_score"].tolist()
                jaccard_weights = df["weight"].tolist()
                df = None

                self.result_write(["Normal", "Anomalies"], distribution_heatmap, "Jaccard Anomaly Community Distribution", "", "", "D")
                self.result_write(jaccard_weights, jaccard_similarities, "Weights-Similarities", "Weights", "Similarities", "S")

                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)
                print("Edges Results Ends")

            if get_anomaly_nodes:
                print("========================\nNodes Results Starts")
                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)

                nodes_set = df_eliminated["source"].tolist() + df_eliminated["target"].tolist()
                nodes_set = set(nodes_set)
                for node in nodes_set:
                    df_node_group = df_eliminated[(df_eliminated["source"] == node) | (df_eliminated["target"] == node)]
                    df_node_group_filtered = df_node_group[(df_node_group["jaccard_anomaly_score"].values > self.edge_anomaly_distance)]
                    if len(df_node_group_filtered) > 0:
                        count_of_all_edges = len(df_node_group)
                        count_of_anomaly_edges = len(df_node_group_filtered)
                        count_of_non_anomaly_edges = count_of_all_edges - count_of_anomaly_edges

                        count_factor_of_anomaly_edges = count_of_anomaly_edges / count_of_all_edges
                        count_factor_of_non_anomaly_edges = count_of_non_anomaly_edges / count_of_all_edges

                        wjFactor = decision_function_nodes(count_factor_of_anomaly_edges)
                        node_anomaly_score = wjFactor * count_factor_of_non_anomaly_edges - count_factor_of_anomaly_edges

                        if node_anomaly_score < 0:
                            print("--------------------")
                            print("Node Id: " + str(node))
                            print("Node Factor: " + str(wjFactor))
                            print("Node Non Anomaly Edge Count: " + str(count_of_non_anomaly_edges))
                            print("Node Anomaly Edge Count: " + str(count_of_anomaly_edges))
                            print("Node Anomaly Score: " + str(node_anomaly_score))

                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print("--------------------")
                print(init_time)
                print("Nodes Results Ends")

            if get_anomaly_communities:
                print("========================\nCommunities Results Starts")
                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)

                graph = nx.from_pandas_edgelist(df_eliminated, "source", "target", ["jaccard_anomaly_score"])
                partition = community_louvain.best_partition(graph, randomize=False)

                nodes_set = df_eliminated["source"].tolist() + df_eliminated["target"].tolist()
                nodes_set = set(nodes_set)

                communities_distribution = []
                for item in nodes_set:
                    community_flag_number = partition[item]
                    communities_distribution.append(community_flag_number)
                distinct_communities_distribution = set(communities_distribution)

                communities_edges_set = []
                for community in distinct_communities_distribution:
                    communities_nodes_set = []
                    for item in partition:
                        if partition[item] == community:
                            communities_nodes_set.append(item)
                    communities_edges_set.append(df_eliminated[df_eliminated["source"].isin(communities_nodes_set) & df_eliminated["target"].isin(communities_nodes_set)])
                df_graph = None

                for index, item in enumerate(communities_edges_set):
                    if len(item.index) == 0:
                        print("--------------------")
                        print("Community: " + str(list(distinct_communities_distribution)[index]))
                        print("Anomaly Score: No Edges")
                    else:
                        community_anomaly_score = round(item["jaccard_anomaly_score"].mean(), self.round_decimals)
                        if community_anomaly_score > self.community_anomaly_distance:
                            print("--------------------")
                            print("Community: " + str(list(distinct_communities_distribution)[index]))
                            print("Anomaly Score: " + str(community_anomaly_score))

                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print("--------------------")
                print(init_time)
                print("Communities Results Ends")

            if len(sub_graph_nodes) != 0:
                print("========================\nQuery Results Starts")
                sub_graph_edges = df_eliminated[df_eliminated["source"].isin(
                    sub_graph_nodes) & df_eliminated["target"].isin(sub_graph_nodes)]
                print("Sub Graph: " + str(list(sub_graph_nodes)))
                if len(sub_graph_edges.index) == 0:
                    print("Anomaly Score: No Edges")
                else:
                    print("Anomaly Score: " + str(
                        round(sub_graph_edges["jaccard_anomaly_score"].mean(), self.round_decimals)))
                    print("Anomaly Threshold: " + str(self.community_anomaly_distance))
                print(init_time)
                print("Query Results Ends")

        except Exception as error:
            print("===================================================================================================")
            print("Error on_main: %s" % str(error))
            print("===================================================================================================")
            traceback.print_exc()
            print("===================================================================================================")
            print(init_time)
            print(end_time)




