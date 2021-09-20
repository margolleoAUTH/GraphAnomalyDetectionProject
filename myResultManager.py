import traceback
import datetime
import pandas as pd
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import community as community_louvain
from pathlib import Path
from collections import defaultdict
from myFileManager import MyFileManager
from sklearn.feature_extraction.text import TfidfVectorizer


class MyResultManager:

    def __init__(self, year_directory_path, save_results):
        self.year_directory_path = year_directory_path
        self.profiles_directory_path = Path(self.year_directory_path.__str__() + "/profiles")
        self.results_directory_path = Path(self.year_directory_path.__str__() + "/results")
        self.save_results = save_results
        self.partition = None
        self.round_decimals = 8
        self.edge_anomaly_distance = 0.45995
        self.file_manager = MyFileManager()
        self.tf_idf_vectorizer = TfidfVectorizer(max_features=500)

    def decision_function(self, total_edges, entity_edges, anomalies_edges):
        anomaly_likelihood = 1 - (2 * self.edge_anomaly_distance)
        entity_likelihood = entity_edges / total_edges
        limit = total_edges * anomaly_likelihood * entity_likelihood
        result = anomalies_edges > limit
        return result

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

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if result_type == "D":
            if len(X) < 2:
                ax = plt.gca()
                ax.set_xlim(-5, 5)
                plt.bar(X, Y[0], color="#9999ff", width=1)
                plt.bar(X, Y[1], color="#ff9999", width=1)
            else:
                plt.subplot(2, 1, 1)
                plt.plot(X, Y[0], color="#9999ff")
                plt.title(title)
                plt.ylabel("Non-Anomalies")
                plt.subplot(2, 1, 2)
                plt.plot(X, Y[1], color="#ff9999")
                plt.ylabel("Anomalies")

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

        if self.save_results:
            f.savefig(self.results_directory_path.__str__() + "/" + title + ".pdf", bbox_inches="tight")
        else:
            plt.show()

    def df_time_reader(self, row, hash_map_nodes, column_type, start_year, end_year, root_directory_path):
        node_id = row[column_type]
        length = hash_map_nodes[node_id]["length"]
        papers_array = []
        for year in range(start_year, end_year + 1):
            path = root_directory_path + "/" + str(year) + "/jsons/" + str(row[column_type]) + ".txt"
            if os.path.isfile(path):
                with open(path) as json_file:
                    json_array = json.load(json_file)
                    for item in json_array:
                        papers_array.append(item["text"])
        self.tf_idf_vectorizer.fit_transform(papers_array)
        feature_array = np.array(self.tf_idf_vectorizer.get_feature_names())
        profile_tf_idf = " ".join(feature_array)
        return pd.Series([profile_tf_idf, length], index=["text_" + column_type, "length_" + column_type])

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

    def df_edge_anomaler_derivative(self, row):
        row = row.tolist()
        row.pop(0)
        row_values = [x for x in row if not math.isnan(x)]
        anomaly = False
        limit = len(row_values) - 1
        for index, element in enumerate(row_values):
            if index != limit:
                current_flag = "OO"
                next_flag = "OO"

                if element > self.edge_anomaly_distance:
                    current_flag = "PA"
                if element < (-1 * self.edge_anomaly_distance):
                    current_flag = "NA"

                next_element = row_values[index + 1]
                if next_element > self.edge_anomaly_distance:
                    next_flag = "PA"
                if next_element < (-1 * self.edge_anomaly_distance):
                    next_flag = "NA"

                if current_flag != next_flag:
                    anomaly = True
                    break
        return anomaly

    def df_node_anomaler_derivative(self, row):
        row = row.tolist()
        row.pop(0)
        row_values = [x for x in row if not math.isnan(x)]
        anomaly = False
        limit = len(row_values) - 1
        for index, element in enumerate(row_values):
            if index != limit:
                current_flag = "NA"
                next_flag = "NA"

                if element < 1:
                    current_flag = "AA"

                next_element = row_values[index + 1]
                if next_element < 1:
                    next_flag = "AA"

                if current_flag != next_flag:
                    anomaly = True
                    break
        return anomaly

    def df_scores_wrapper(self, df):
        print("========================\nScores Calculation Starts")
        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)

        df_weight = df.apply(self.df_weighter, axis=1)
        df_weight = df_weight.astype("category")
        df_jaccard = df.apply(self.df_jaccarder, axis=1)
        df_jaccard = df_jaccard.astype("category")
        df = pd.concat([df["id"], df["source"], df["target"], df_jaccard, df_weight], axis=1, join="inner")
        df = df.rename(columns={0: "jaccard", 1: "weight"})
        df = df.astype("category")
        df_weight = None
        df_jaccard = None

        df_prime = df.apply(self.df_anomaler, axis=1)
        df_eliminated = df_prime[["id", "source", "target", "jaccard_anomaly_score"]]

        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)
        print("Scores Calculation Ends")

        return [df_prime, df_eliminated]

    def df_edges_wrapper(self, df):
        print("========================\nEdges Results Starts")
        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)

        del df["source"]
        del df["target"]
        hash_map_jaccard_anomalies = df.to_dict("index")
        self.file_manager.dict_csv_write(self.results_directory_path.__str__() + "/jacardAnomalies.csv",
                                         ["id", "jaccard_anomaly_score", "jaccard_similarity_score", "weight"],
                                         hash_map_jaccard_anomalies)

        jaccard_anomaly_count_pos = df["jaccard_anomaly_score"][
            df["jaccard_anomaly_score"] > self.edge_anomaly_distance
            ].count()
        jaccard_anomaly_count_neg = df["jaccard_anomaly_score"][
            df["jaccard_anomaly_score"] < -1 * self.edge_anomaly_distance
            ].count()
        jaccard_non_anomaly_count_pos = df["jaccard_anomaly_score"][
            (df["jaccard_anomaly_score"] <= self.edge_anomaly_distance) & (df["jaccard_anomaly_score"] >= 0.0)
            ].count()
        jaccard_non_anomaly_count_neg = df["jaccard_anomaly_score"][
            (df["jaccard_anomaly_score"] >= -1 * self.edge_anomaly_distance) & (df["jaccard_anomaly_score"] < 0.0)
            ].count()
        distribution_heatmap = [
            [jaccard_non_anomaly_count_pos, jaccard_anomaly_count_pos],
            [jaccard_non_anomaly_count_neg, jaccard_anomaly_count_neg]
        ]
        jaccard_similarities = df["jaccard_similarity_score"].tolist()
        jaccard_weights = df["weight"].tolist()

        self.result_write(["Normal", "Anomalies"], distribution_heatmap, "Jaccard Anomaly Distribution", "", "",
                          "H")
        self.result_write(jaccard_weights, jaccard_similarities, "Weights-Similarities", "Weights",
                          "Similarities", "S")

        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)
        print("Edges Results Ends")

    def df_nodes_wrapper(self, df_eliminated):
        print("========================\nNodes Results Starts")
        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)

        count_of_total_edges = len(df_eliminated.index)

        myfile = open(self.results_directory_path.__str__() + "\\NodeResults.txt", "w")

        nodes_set = df_eliminated["source"].tolist() + df_eliminated["target"].tolist()
        nodes_set = set(nodes_set)
        dfn = pd.DataFrame()
        for node in nodes_set:
            df_node_group = df_eliminated[(df_eliminated["source"] == node) | (df_eliminated["target"] == node)]
            df_node_group_filtered = df_node_group[
                (df_node_group["jaccard_anomaly_score"].values > self.edge_anomaly_distance) | (
                        df_node_group["jaccard_anomaly_score"].values < -1 * self.edge_anomaly_distance)]

            count_of_all_node_edges = len(df_node_group)
            count_of_anomaly_edges = len(df_node_group_filtered)
            count_of_non_anomaly_edges = count_of_all_node_edges - count_of_anomaly_edges
            node_anomaly_score = self.decision_function(count_of_total_edges, count_of_all_node_edges,
                                                             count_of_anomaly_edges)

            year = self.year_directory_path.__str__().split("\\")
            year = year[len(year) - 1]
            dfn = dfn.append({"Id": str(node), year: node_anomaly_score}, ignore_index=True)

            if node_anomaly_score:
                myfile.write("----------------------------------------" + "\n")
                myfile.write("Node Id: " + str(node) + "\n")
                myfile.write("Node Edge Count: " + str(count_of_all_node_edges) + "\n")
                myfile.write("Node Non Anomaly Edge Count: " + str(count_of_non_anomaly_edges) + "\n")
                myfile.write("Node Anomaly Edge Count: " + str(count_of_anomaly_edges) + "\n")
                myfile.write("Node Anomaly Score: " + str(node_anomaly_score) + "\n")

        myfile.close()
        dfn.to_hdf(self.results_directory_path.__str__() + "/dfn.h5", "dfn", format="table")
        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)
        print("Nodes Results Ends")

    def df_community_wrapper(self, df_eliminated, community_type):
        print("========================\nCommunities Results Starts")
        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)

        count_of_total_edges = len(df_eliminated.index)

        communities_edges_set = []
        hash_map_community_members = defaultdict(list)
        for key, value in sorted(self.partition.items()):
            hash_map_community_members[value].append(key)
        for item in hash_map_community_members:
            community_members = hash_map_community_members[item]
            communities_edges_set.append(df_eliminated[
                                             df_eliminated["source"].isin(community_members) & df_eliminated[
                                                 "target"].isin(community_members)])

        result = self.df_community(communities_edges_set, hash_map_community_members, count_of_total_edges, community_type)

        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)
        print("Communities Results Ends")
        if not math.isnan(community_type):
            return result

    def df_community(self, communities_edges_set, hash_map_community_members, count_of_total_edges, community_type):
        to_return = False
        if not math.isnan(community_type):
            to_return = True
            community_type = str(community_type) + "-Community"

        myfile = open(self.results_directory_path.__str__() + "\\" + community_type + "Results.txt", "w")

        count_of_anomaly_communities = 0
        community_distribution = 0
        anomaly_community_distribution = []
        non_anomaly_community_distribution = []

        for index, item in enumerate(communities_edges_set):
            community = str(list(hash_map_community_members.keys())[index])
            if len(item.index) == 0:
                myfile.write("----------------------------------------" + "\n")
                myfile.write(community_type + ": " + community + "\n")
                myfile.write("Anomaly Score: No Edges" + "\n")
            else:
                count_of_community_edges = len(item.index)
                count_of_anomaly_edges = item["jaccard_anomaly_score"][
                    (item["jaccard_anomaly_score"] > self.edge_anomaly_distance) | (
                            item["jaccard_anomaly_score"] < -1 * self.edge_anomaly_distance)].count()
                community_anomaly_score = self.decision_function(count_of_total_edges, count_of_community_edges, count_of_anomaly_edges)
                count_of_non_anomaly_edges = count_of_community_edges - count_of_anomaly_edges
                if count_of_anomaly_edges > 0:
                    community_distribution += 1
                    anomaly_community_distribution.append(-1 * count_of_anomaly_edges)
                    non_anomaly_community_distribution.append(count_of_non_anomaly_edges)
                    item.to_csv(self.results_directory_path.__str__() + "\\" + str(
                        community) + "-" + community_type + "Results.csv", index=False, header=True)
                    myfile.write("----------------------------------------" + "\n")
                    myfile.write(community_type + ": " + community + "\n")
                    myfile.write("Anomaly Score: " + str(community_anomaly_score) + "\n")

                if community_anomaly_score:
                    count_of_anomaly_communities = count_of_anomaly_communities + 1

        myfile.close()
        if community_distribution > 0:
            community_distribution = list(range(community_distribution))
            distribution_set = [non_anomaly_community_distribution, anomaly_community_distribution]
            self.result_write(community_distribution, distribution_set, "Jaccard Anomaly " + community_type +
                              " Distribution", "", "", "D")

        if to_return:
            return count_of_anomaly_communities

    def manipulate_results(self,
                           get_anomaly_edges=False,
                           get_anomaly_nodes=False,
                           get_anomaly_communities=False,
                           sub_graph_nodes=None):

        if sub_graph_nodes is None:
            sub_graph_nodes = []

        try:
            self.file_manager.directory_manipulation(self.results_directory_path)
            df = pd.read_csv(self.year_directory_path.__str__() + "/gephiNodes.csv")
            hash_map_nodes = df.set_index("id").T.to_dict()
            df = pd.read_csv(self.year_directory_path.__str__() + "/gephiEdges.csv")
            df = df.astype("category")

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

            df_scores = self.df_scores_wrapper(df)
            df_prime = df_scores[0]
            df_eliminated = df_scores[1]

            if get_anomaly_edges:
                self.df_edges_wrapper(df_prime)

            if get_anomaly_nodes:
                self.df_nodes_wrapper(df_eliminated)

            if get_anomaly_communities:
                print("========================\nCommunities Results Starts")
                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)

                graph = nx.from_pandas_edgelist(df_eliminated, "source", "target", ["jaccard_anomaly_score"])
                self.partition = community_louvain.best_partition(graph, randomize=False)
                self.df_community_wrapper(df_eliminated, "jaccard_anomaly_score", "Community")

                init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
                print(init_time)
                print("Communities Results Ends")

            # if len(sub_graph_nodes) != 0:
            #     df = df[df["source"].isin(sub_graph_nodes) | df["target"].isin(sub_graph_nodes)]
            #     df_eliminated = df_eliminated[df_eliminated["source"].isin(sub_graph_nodes) | df_eliminated["target"].isin(sub_graph_nodes)]
            #     print("========================\nQuery Results Starts")
            #     init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            #     print(init_time)
            #
            #     sub_graph_edges = df_eliminated[df_eliminated["source"].isin(
            #         sub_graph_nodes) & df_eliminated["target"].isin(sub_graph_nodes)]
            #     communities_edges_set = [sub_graph_edges]
            #     hash_map_community_members = defaultdict(list)
            #     hash_map_community_members[0].append(sub_graph_nodes)
            #
            #     self.df_community(communities_edges_set, hash_map_community_members, "Sub Graph")
            #
            #     init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            #     print(init_time)
            #     print("Query Results Ends")

        except Exception as error:
            print("===================================================================================================")
            print("Error: %s" % str(error))
            print("===================================================================================================")
            traceback.print_exc()
            print("===================================================================================================")

    def update_time_results(self, start_year, end_year):
        print("========================\nResults Preparation Starts")
        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)
        root_directory_path = self.year_directory_path.__str__()

        self.year_directory_path = Path(root_directory_path + "/" + str(start_year))
        self.profiles_directory_path = Path(self.year_directory_path.__str__() + "/profiles")
        self.file_manager.directory_manipulation(self.results_directory_path)

        nodes = pd.read_csv(self.year_directory_path.__str__() + "/gephiNodes.csv")
        edges = pd.read_csv(self.year_directory_path.__str__() + "/gephiEdges.csv")

        hash_map_nodes = nodes.set_index("id").T.to_dict()
        df = edges.astype("category")

        df_source = df.apply(lambda x: self.df_time_reader(x, hash_map_nodes, "source", start_year, start_year, root_directory_path), axis=1)
        df_source = df_source.astype("category")
        df_target = df.apply(lambda x: self.df_time_reader(x, hash_map_nodes, "target", start_year, start_year, root_directory_path), axis=1)
        df_target = df_target.astype("category")
        hash_map_nodes = None

        df = pd.concat([df["id"], df["source"], df["target"], df_source, df_target, df["length"]], axis=1, join="inner")
        df = df.astype("category")
        df_source = None
        df_target = None
        df.to_hdf(self.results_directory_path.__str__() + "/df_" + str(start_year) + "_" + str(start_year) + ".h5", "df", format="table")

        for year in range(start_year + 1, end_year + 1):
            self.year_directory_path = Path(root_directory_path + "/" + str(year))
            self.profiles_directory_path = Path(self.year_directory_path.__str__() + "/profiles")

            temp_nodes = pd.read_csv(self.year_directory_path.__str__() + "/gephiNodes.csv")
            temp_edges = pd.read_csv(self.year_directory_path.__str__() + "/gephiEdges.csv")

            nodes = pd.concat([nodes.set_index("id"), temp_nodes.set_index("id")]).reset_index()
            nodes = nodes.groupby(["id", "name"]).sum().reset_index()
            edges = pd.concat([edges.set_index("id"), temp_edges.set_index("id")]).reset_index()
            edges = edges.groupby(["id", "source", "target"]).sum().reset_index()

            # duplicate = edges[edges.index.duplicated(keep=False)]
            # res1 = edges.loc[edges["id"] == "f91fc7021139f2b23ed1781d16479ce934b57998"]
            # res2 = temp_edges.loc[temp_edges["id"] == "f91fc7021139f2b23ed1781d16479ce934b57998"]
            # res3 = edges.loc[edges["id"] == "f91fc7021139f2b23ed1781d16479ce934b57998"]

            hash_map_nodes = nodes.set_index("id").T.to_dict()
            df = edges.astype("category")

            df_source = df.apply(lambda x: self.df_time_reader(x, hash_map_nodes, "source", start_year, year, root_directory_path), axis=1)
            df_source = df_source.astype("category")
            df_target = df.apply(lambda x: self.df_time_reader(x, hash_map_nodes, "target", start_year, year, root_directory_path), axis=1)
            df_target = df_target.astype("category")
            hash_map_nodes = None

            df = pd.concat([df["id"], df["source"], df["target"], df_source, df_target, df["length"]], axis=1,
                           join="inner")
            df = df.astype("category")
            df_source = None
            df_target = None
            df.to_hdf(self.results_directory_path.__str__() + "/df_" + str(start_year) + "_" + str(year) + ".h5", "df", format="table")

        init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
        print(init_time)
        print("Results Preparation Ends")

    def manipulate_time_results(self, start_year, end_year):
        root_results_directory_path = self.results_directory_path.__str__()
        count_of_anomaly_communities_list = []
        df_results_eliminated = pd.DataFrame()

        for year in range(start_year, end_year + 1):
            self.results_directory_path = Path(root_results_directory_path + "/" + str(year))
            self.file_manager.directory_manipulation(self.results_directory_path)

            df = pd.read_hdf(root_results_directory_path + "/df_" + str(start_year) + "_" + str(year) + ".h5", "df")

            df_scores = self.df_scores_wrapper(df)
            df_prime = df_scores[0]
            df_eliminated = df_scores[1]

            self.df_edges_wrapper(df_prime)
            self.df_nodes_wrapper(df_eliminated)

            graph = nx.from_pandas_edgelist(df_eliminated, "source", "target", ["jaccard_anomaly_score"])
            self.partition = community_louvain.best_partition(graph, randomize=False)
            count_of_anomaly_communities = self.df_community_wrapper(df_eliminated, year)
            count_of_anomaly_communities_list.append(count_of_anomaly_communities)

            del df_eliminated["source"]
            del df_eliminated["target"]
            if df_results_eliminated.empty:
                df_results_eliminated = df_eliminated
            else:
                df_results_eliminated = df_results_eliminated.merge(df_eliminated, on="id", how="outer")
            df_results_eliminated.columns = [*df_results_eliminated.columns[:-1], str(year)]

        df = df_results_eliminated.apply(self.df_edge_anomaler_derivative, axis=1)
        result = pd.concat([df_results_eliminated, df], axis=1)
        result = result.rename(columns={0: "result"})
        result = result.query("result")
        self.file_manager.df_csv_write(self.year_directory_path.__str__() + "/edges_results.csv", result)

        dfn = pd.read_hdf(root_results_directory_path + "/" + str(start_year) + "/dfn.h5", "dfn")
        dfn.columns = [*dfn.columns[:-1], str(start_year)]
        for year in range(start_year + 1, end_year + 1):
            temp_dfn = pd.read_hdf(root_results_directory_path + "/" + str(year) + "/dfn.h5", "dfn")
            dfn = dfn.merge(temp_dfn, on="Id", how="outer")
            dfn.columns = [*dfn.columns[:-1], str(year)]

        columns = list(range(start_year, end_year + 1))
        columns = [str(i) for i in columns]
        columns.insert(0, "Id")
        dfn = dfn.reindex(columns=columns)

        df = dfn.apply(self.df_node_anomaler_derivative, axis=1)
        result = pd.concat([dfn, df], axis=1)
        result = result.rename(columns={0: "result"})
        result = result.query("result")
        self.file_manager.df_csv_write(self.year_directory_path.__str__() + "/nodes_results.csv", result)

        columns = list(range(start_year, end_year + 1))
        columns = [str(i) for i in columns]
        columns.insert(0, "Id")
        count_of_anomaly_communities_list.insert(0, "Graph")
        result = pd.DataFrame([count_of_anomaly_communities_list], columns=columns)
        self.file_manager.df_csv_write(self.year_directory_path.__str__() + "/communities_results.csv", result)

        # data = {33781-506-130
        #     "id": ["0", "1", "2"],
        #     "source": [0, 1, 1],
        #     "target": [1, 2, 3],
        #     "weight": [0.1, 0.3, 0.2]
        # }
        # df = pd.DataFrame(data)
        # print(df)
        # G = nx.from_pandas_edgelist(df, edge_attr=["source", "target", "weight"])
        #
        # data = {
        #     "id": ["3", "4", "2"],
        #     "source": [0, 1, 1],
        #     "target": [2, 4, 3],
        #     "weight": [0.8, 0.8, 0.5]
        # }
        # temp_df = pd.DataFrame(data)
        # # G1 = nx.from_pandas_edgelist(temp_df, edge_attr=["source", "target", "weight"])
        # # G.update(G1)
        #
        # # df_Graph = df.merge(temp_df, on="id", how="outer")
        # df_Graph = pd.concat([df.set_index("id"), temp_df.set_index("id")])
        # df_Graph = df_Graph[~df_Graph.index.duplicated(keep="last")].reset_index()
        #
        # G = nx.from_pandas_edgelist(df_Graph, edge_attr=["source", "target", "weight"])
        #
        # pos = nx.spring_layout(G)
        # nx.draw(G, pos, with_labels=True)
        # labels = nx.get_edge_attributes(G, "weight")
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        # plt.show()
