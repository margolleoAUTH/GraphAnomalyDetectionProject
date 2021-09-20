import traceback
import datetime
import dateutil.parser
import pandas as pd
import json
import numpy as np
import os
import hashlib
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from myPreProcessor import MyPreProcessor


def my_hash(value):
    encoded = value.encode()
    result = hashlib.sha1(encoded)
    return result.hexdigest()


class MyGraphManager:

    def __init__(self,
                 year_directory_path,
                 global_metadata_directory,
                 global_metadata_file,
                 top_n_tf_idf_limit,
                 year,
                 file_manager):
        self.year = year
        self.year_directory_path = year_directory_path
        self.jsons_directory_path = Path(self.year_directory_path.__str__() + "/jsons")
        self.profiles_directory_path = Path(self.year_directory_path.__str__() + "/profiles")
        self.global_metadata_directory = global_metadata_directory
        self.global_metadata_file = global_metadata_file
        self.tf_idf_vectorizer = TfidfVectorizer(max_features=top_n_tf_idf_limit)
        self.pre_processor = MyPreProcessor()
        self.file_manager = file_manager

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

    def manipulate_data(self):
        init_time = None
        end_time = None

        hash_map_nodes = {}
        hash_map_edges = {}

        try:
            self.file_manager.directory_manipulation(self.jsons_directory_path)

            df = pd.read_csv(self.global_metadata_directory + "/" + self.global_metadata_file)

            df = df[(pd.notna(df["publish_time"]))]
            df = df[["publish_time", "pdf_json_files", "pmc_json_files"]]

            df["publish_time"] = df["publish_time"].apply(lambda x: dateutil.parser.parse(str(x)).year)
            df_unfiltered_index = str(len(df.index))

            df_filtered = df[(pd.notna(df["pdf_json_files"])) | (pd.notna(df["pmc_json_files"]))]
            df_filtered_index = str(len(df_filtered.index))

            if "_" in self.year:
                years = self.year.split("_")
                start_year = int(years[0])
                end_year = int(years[1])
                years = list(range(start_year, end_year + 1))
            else:
                years = [int(self.year)]

            grouped_df = df_filtered[df_filtered["publish_time"].isin(years)]
            df_grouped_index = str(len(grouped_df.index))

            paper_iteration_index = 0

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

                with open(self.global_metadata_directory + "/" + pdf_json_file) as pdf_file, open(self.global_metadata_directory + "/" + pmc_json_file) as pmc_file:
                    data_pdf = json.load(pdf_file)
                    data_pcm = json.load(pmc_file)

                    authors = data_pdf["metadata"]["authors"]
                    if len(authors) == 0:
                        authors = data_pcm["metadata"]["authors"]

                    paper_text_clear = ""
                    if len(authors) != 0:

                        paper_text = data_pdf["metadata"]["title"]
                        if len(paper_text) == 0:
                            paper_text = data_pcm["metadata"]["title"]

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
                            author_text_clear = self.data_manipulation(author_text_clear, "").strip()
                            if author_text_clear == "nan":
                                author_text_clear = "not_a_name"
                            if author_text_clear != "" and (author_text_clear not in authors_as_list):
                                authors_as_list.append(author_text_clear)
                                key = my_hash(author_text_clear)
                                if key not in hash_map_nodes:
                                    hash_map_nodes[key] = {
                                        "id": key,
                                        "name": author_text_clear,
                                        "length": len(paper_text_clear),
                                        "papers_array": [{"paper_id_pdf": data_pdf["paper_id"], "paper_id_pcm": data_pcm["paper_id"], "text": paper_text_clear}]
                                    }
                                    with open(self.jsons_directory_path.__str__() + "/" + key + ".txt", "w") as json_file:
                                        json.dump(hash_map_nodes[key]["papers_array"], json_file)
                                else:
                                    hash_map_nodes[key]["length"] += len(paper_text_clear)
                                    hash_map_nodes[key]["papers_array"].append({"paper_id_pdf": data_pdf["paper_id"], "paper_id_pcm": data_pcm["paper_id"], "text": paper_text_clear})
                                    with open(self.jsons_directory_path.__str__() + "/" + key + ".txt", "w") as json_file:
                                        json.dump(hash_map_nodes[key]["papers_array"], json_file)

                            print("Status: " + "Authors(Nodes) Paper's Prepared")

                        if len(authors_as_list) == 1:

                            print("Status: " + "Author-Pairs Number is 1")

                        else:

                            authors_as_list.sort()
                            authors_r_subset = list(combinations(authors_as_list, 2))
                            print("Status: " + "Author-Pairs Number is " + str(len(authors_r_subset)))

                            for authors_pair in authors_r_subset:
                                hashed_author_source = my_hash(authors_pair[0])
                                hashed_author_target = my_hash(authors_pair[1])
                                if hashed_author_source != hashed_author_target:
                                    key = my_hash(str(hashed_author_source) + str(hashed_author_target))
                                    if key not in hash_map_edges:
                                        hash_map_edges[key] = {
                                            "id": key,
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

            df_hash_map_nodes = pd.DataFrame.from_dict(hash_map_nodes, orient="index").T
            df_hash_map_nodes = df_hash_map_nodes.drop("papers_array")
            hash_map_nodes = df_hash_map_nodes.to_dict()

            self.file_manager.dict_csv_write(self.year_directory_path.__str__() + "/gephiNodes.csv", ["id", "name", "length"], hash_map_nodes)
            self.file_manager.dict_csv_write(self.year_directory_path.__str__() + "/gephiEdges.csv", ["id", "source", "target", "weight", "length", "undirected"], hash_map_edges)
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
            print("Error: %s" % str(error))
            print("===================================================================================================")
            traceback.print_exc()
            print("===================================================================================================")
            print(init_time)
            print(end_time)

    def manipulate_profiles(self):
        init_time = None
        end_time = None

        try:
            self.file_manager.directory_manipulation(self.profiles_directory_path)

            init_time = datetime.datetime.now().strftime("%D %H:%M:%S")
            files = os.listdir(self.jsons_directory_path.__str__())
            for key in files:
                with open(self.jsons_directory_path.__str__() + "/" + key) as json_file:
                    json_array = json.load(json_file)
                    # if json_array[0]["text"] == "":
                    #     profile_tf_idf = ""
                    # else:
                    papers_array = []
                    for item in json_array:
                        papers_array.append(item["text"])
                    self.tf_idf_vectorizer.fit_transform(papers_array)
                    feature_array = np.array(self.tf_idf_vectorizer.get_feature_names())
                    profile_tf_idf = " ".join(feature_array)
                    with open(self.profiles_directory_path.__str__() + "/" + key, "w") as profile_file:
                        json.dump({"text": profile_tf_idf}, profile_file)
            end_time = datetime.datetime.now().strftime("%D %H:%M:%S")

            print("========================\nmanipulate_profiles: Run time details")
            print(init_time)
            print(end_time)
            print("manipulate_profiles: Run time details\n========================")

        except Exception as error:
            print("===================================================================================================")
            print("Error: %s" % str(error))
            print("===================================================================================================")
            traceback.print_exc()
            print("===================================================================================================")
            print(init_time)
            print(end_time)



