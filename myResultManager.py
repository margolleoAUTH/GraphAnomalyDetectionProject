import traceback
import datetime
import pandas as pd
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import community as community_louvain
from pathlib import Path
from collections import defaultdict
from myPreProcessor import MyPreProcessor
from myFileManager import MyFileManager


class MyResultManager:

    def __init__(self, year_directory_path, save_results, read_csv_directory):
        self.year_directory_path = year_directory_path
        self.profiles_directory_path = Path(self.year_directory_path.__str__() + "/profiles")
        self.results_directory_path = Path(self.year_directory_path.__str__() + "/results")
        self.save_results = save_results
        self.read_csv_directory = read_csv_directory
        self.round_decimals = 8
        self.edge_anomaly_space = 0.0228
        self.edge_anomaly_distance = 0.50 - self.edge_anomaly_space
        self.pre_processor = MyPreProcessor()
        self.file_manager = MyFileManager()

    def decision_function(self, non_anomalies, anomalies):
        total = non_anomalies + anomalies
        non_anomalies_score = non_anomalies / total
        anomalies_score = anomalies / total
        factor = 1 / pow(10, anomalies_score)
        result = non_anomalies_score * factor - anomalies_score
        return round(result, self.round_decimals)

    def jaccard_similarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def get_jaccard_similarity_between_two_strings(self, strings_array):
        vector0 = strings_array[0].split(" ")
        vector1 = strings_array[1].split(" ")
        similarity_score = self.jaccard_similarity(vector0, vector1)
        return round(similarity_score, self.round_decimals)

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
            if len(X) < 2:
                ax = plt.gca()
                ax.set_xlim(-5, 5)
                plt.bar(X, Y[0], color="#9999ff", width=1)
                plt.bar(X, Y[1], color="#ff9999", width=1)
            else:
                plt.plot(X, Y[0], color="#9999ff")
                plt.plot(X, Y[1], color="#ff9999")

        if result_type == "H":
            data = pd.DataFrame(Y, columns=X, index=["Positive", "Negative"])
            sns.heatmap(data, annot=True)

        if result_type == "S":
            plt.scatter(X, Y)
            x = np.array([0, 1])
            y = np.array([0, 1])
            plt.plot(x, y, color="r")

            x1 = np.array([0, 1 - self.edge_anomaly_distance])
            y1 = np.array([self.edge_anomaly_distance, 1])
            plt.plot(x1, y1, linestyle="dotted", color="c")
            x2 = np.array([self.edge_anomaly_distance, 1])
            y2 = np.array([0, 1 - self.edge_anomaly_distance])
            plt.plot(x2, y2, linestyle="dotted", color="c")

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if self.save_results:
            f.savefig(self.results_directory_path.__str__() + "/" + title + ".pdf", bbox_inches="tight")
        else:
            plt.show()

    def df_reader(self, row, hash_map_nodes, column_type):
        node_id = row[column_type]
        length = hash_map_nodes[node_id]["length"]
        path = self.profiles_directory_path.__str__() + "/" + str(row[column_type]) + ".txt"
        with open(path) as profile_file:
            json_profile = json.load(profile_file)
            json_profile_text = json_profile["text"]
        return pd.Series([json_profile_text, length], index=["text_" + column_type, "length_" + column_type])

    def df_weighter(self, row):
        total = row["length_source"] + row["length_target"] - row["length"] + 1
        return round(row["length"] / total, self.round_decimals)

    def df_jaccarder(self, row):
        return self.get_jaccard_similarity_between_two_strings([row["text_source"], row["text_target"]])

    def df_anomaler(self, row):
        jaccard_anomaly_score = row["jaccard"] - row["weight"]
        return pd.Series([row["id"], row["source"], row["target"], jaccard_anomaly_score, row["jaccard"], row["weight"]
                          ], index=["id", "source", "target", "jaccard_anomaly_score", "jaccard_similarity_score",
                                    "weight"])

    def df_community(self, communities_edges_set, hash_map_community_members, community_type):
        myfile = open(self.results_directory_path.__str__() + "\\" + community_type + "Results.txt", "w")

        community_distribution = 0
        anomaly_community_distribution = []
        non_anomaly_community_distribution = []

        # print("Sub Graph: " + str(list(sub_graph_nodes)))
        # if len(sub_graph_edges.index) == 0:
        #     print("Anomaly Score: No Edges")
        # else:
        #     print("Anomaly Score: " + str(
        #         round(sub_graph_edges["jaccard_anomaly_score"].mean(), self.round_decimals)))
        #     print("Anomaly Threshold: " + str(self.community_anomaly_distance))

        for index, item in enumerate(communities_edges_set):
            community = str(list(hash_map_community_members.keys())[index])
            if len(item.index) == 0:
                myfile.write("----------------------------------------" + "\n")
                myfile.write(community_type + ": " + community + "\n")
                myfile.write("Anomaly Score: No Edges" + "\n")
            else:
                total = len(item.index)
                count_of_anomaly_edges = item["jaccard_anomaly_score"][
                    (item["jaccard_anomaly_score"] > self.edge_anomaly_distance) | (
                            item["jaccard_anomaly_score"] < -1 * self.edge_anomaly_distance)].count()
                count_of_non_anomaly_edges = total - count_of_anomaly_edges
                community_anomaly_score = self.decision_function(count_of_non_anomaly_edges, count_of_anomaly_edges)
                if count_of_anomaly_edges > 0:
                    community_distribution += 1
                    anomaly_community_distribution.append(-1 * count_of_anomaly_edges)
                    non_anomaly_community_distribution.append(count_of_non_anomaly_edges)
                if community_anomaly_score < 0:
                    myfile.write("----------------------------------------" + "\n")
                    myfile.write(community_type + ": " + community + "\n")
                    myfile.write("Anomaly Score: " + str(community_anomaly_score) + "\n")

        if community_distribution > 0:
            community_distribution = list(range(community_distribution))
            distribution_set = [non_anomaly_community_distribution, anomaly_community_distribution]
            self.result_write(community_distribution, distribution_set, "Jaccard Anomaly " + community_type +
                              " Distribution", "", "", "D")

        myfile.close()

    def df_concatenate_years(self, tail):
        all_files = glob.glob(self.year_directory_path.__str__() + tail)
        all_files_list = []
        for filename in all_files:
            all_files_list.append(pd.read_csv(filename))
        df = pd.concat(all_files_list, axis=0, ignore_index=True)
        return df

    def manipulate_results(self,
                           get_anomaly_edges=False,
                           get_anomaly_nodes=False,
                           get_anomaly_communities=False,
                           sub_graph_nodes=None):

        if sub_graph_nodes is None:
            sub_graph_nodes = []

        init_time = None
        end_time = None

        try:
            self.file_manager.directory_manipulation(self.results_directory_path)

            init_time = None
            end_time = None

            print("========================\nScores Calculation Starts")
            init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            print(init_time)

            if self.read_csv_directory is not None:
                df = self.df_concatenate_years("/gephiNodes.csv")
                hash_map_nodes = df.set_index("id").T.to_dict()
                df = self.df_concatenate_years("/gephiEdges.csv")
                df = df.astype("category")
            else:
                df = pd.read_csv(self.year_directory_path.__str__() + "/gephiNodes.csv")
                hash_map_nodes = df.set_index("id").T.to_dict()
                df = pd.read_csv(self.year_directory_path.__str__() + "/gephiEdges.csv")
                df = df.astype("category")

            # df = df.iloc[100:200]
            # df = df.iloc[56000:87000]

            df_source = df.apply(lambda x: self.df_reader(x, hash_map_nodes, "source"), axis=1)
            df_source = df_source.astype("category")
            df_target = df.apply(lambda x: self.df_reader(x, hash_map_nodes, "target"), axis=1)
            df_target = df_target.astype("category")
            hash_map_nodes = None

            df = pd.concat([df["id"], df["source"], df["target"], df_source, df_target, df["length"]], axis=1,
                           join="inner")
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
            df_weight = None
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
                self.file_manager.csv_write(self.year_directory_path.__str__() + "/results/jacardAnomalies.csv",
                                            ["id", "jaccard_anomaly_score", "jaccard_similarity_score", "weight"],
                                            hash_map_jaccard_anomalies)

                jaccard_anomaly_count_pos = df["jaccard_anomaly_score"][
                    df["jaccard_anomaly_score"] > self.edge_anomaly_distance].count()
                jaccard_anomaly_count_neg = df["jaccard_anomaly_score"][
                    df["jaccard_anomaly_score"] < -1 * self.edge_anomaly_distance].count()
                jaccard_non_anomaly_count_pos = df["jaccard_anomaly_score"][
                    (df["jaccard_anomaly_score"] <= self.edge_anomaly_distance) & (
                                df["jaccard_anomaly_score"] > 0.0)].count()
                jaccard_non_anomaly_count_neg = df["jaccard_anomaly_score"][
                    (df["jaccard_anomaly_score"] > -1 * self.edge_anomaly_distance) & (
                                df["jaccard_anomaly_score"] <= 0.0)].count()
                distribution_heatmap = [
                    [jaccard_non_anomaly_count_pos, jaccard_anomaly_count_pos],
                    [jaccard_non_anomaly_count_neg, jaccard_anomaly_count_neg]
                ]
                jaccard_similarities = df["jaccard_similarity_score"].tolist()
                jaccard_weights = df["weight"].tolist()
                df = None

                self.result_write(["Normal", "Anomalies"], distribution_heatmap, "Jaccard Anomaly Distribution", "", "",
                                  "H")
                self.result_write(jaccard_weights, jaccard_similarities, "Weights-Similarities", "Weights",
                                  "Similarities", "S")

                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)
                print("Edges Results Ends")

            if get_anomaly_nodes:
                print("========================\nNodes Results Starts")
                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)
                myfile = open(self.results_directory_path.__str__() + "\\NodeResults.txt", "w")

                nodes_set = df_eliminated["source"].tolist() + df_eliminated["target"].tolist()
                nodes_set = set(nodes_set)
                for node in nodes_set:
                    df_node_group = df_eliminated[(df_eliminated["source"] == node) | (df_eliminated["target"] == node)]
                    df_node_group_filtered = df_node_group[
                        (df_node_group["jaccard_anomaly_score"].values > self.edge_anomaly_distance) | (
                                    df_node_group["jaccard_anomaly_score"].values < -1 * self.edge_anomaly_distance)]
                    if len(df_node_group_filtered) > 0:
                        count_of_all_edges = len(df_node_group)
                        count_of_anomaly_edges = len(df_node_group_filtered)
                        count_of_non_anomaly_edges = count_of_all_edges - count_of_anomaly_edges
                        node_anomaly_score = self.decision_function(count_of_non_anomaly_edges, count_of_anomaly_edges)
                        if node_anomaly_score < 0:
                            myfile.write("----------------------------------------" + "\n")
                            myfile.write("Node Id: " + str(node) + "\n")
                            myfile.write("Node Non Anomaly Edge Count: " + str(count_of_non_anomaly_edges) + "\n")
                            myfile.write("Node Anomaly Edge Count: " + str(count_of_anomaly_edges) + "\n")
                            myfile.write("Node Anomaly Score: " + str(node_anomaly_score) + "\n")

                myfile.close()
                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)
                print("Nodes Results Ends")

            if get_anomaly_communities:
                print("========================\nCommunities Results Starts")
                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)

                graph = nx.from_pandas_edgelist(df_eliminated, "source", "target", ["jaccard_anomaly_score"])
                partition = community_louvain.best_partition(graph, randomize=False)

                communities_edges_set = []
                hash_map_community_members = defaultdict(list)
                for key, value in sorted(partition.items()):
                    hash_map_community_members[value].append(key)
                for item in hash_map_community_members:
                    community_members = hash_map_community_members[item]
                    communities_edges_set.append(df_eliminated[
                                                     df_eliminated["source"].isin(community_members) & df_eliminated[
                                                         "target"].isin(community_members)])

                self.df_community(communities_edges_set, hash_map_community_members, "Community")

                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)
                print("Communities Results Ends")

            if len(sub_graph_nodes) != 0:
                print("========================\nQuery Results Starts")
                sub_graph_edges = df_eliminated[df_eliminated["source"].isin(
                    sub_graph_nodes) & df_eliminated["target"].isin(sub_graph_nodes)]
                communities_edges_set = [sub_graph_edges]
                hash_map_community_members = defaultdict(list)
                hash_map_community_members[0].append(sub_graph_nodes)
                self.df_community(communities_edges_set, hash_map_community_members, "Sub Graph")
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
