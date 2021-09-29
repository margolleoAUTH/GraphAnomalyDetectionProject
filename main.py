import traceback
import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from myGraphManager import MyGraphManager
from myFileManager import MyFileManager
from myResultManager import MyResultManager
from fitter import Fitter, get_common_distributions, get_distributions


if __name__ == "__main__":

    execution_configuration = {
        "JSONs": "1",
        "Profiles": "1",
        "Results": "1",
        "Duration": "1"
    }
    global_metadata_file = "metadata.csv"
    sub_graph_nodes = [
        "22994602fdec19cc9f4e544c6e8aa620c1a0e978",
        "27db0ad1d30a47aedfc26db3ac9102d4316dcbea",
        "22994602fdec19cc9f4e544c6e8aa620c1a0e978",
        "d3db6999267071576b25be7a4ba23abace98b166",
        "d416058f420858d09785114a5ff4334b1566e624",
        "12363b855f9179d4cd99072c0b5cf524436e58c5",
        "9e263a61ea9e247a6c5a5f7b51ccb09ca3bfe57d",
        "27db0ad1d30a47aedfc26db3ac9102d4316dcbea",

    ]
    try:
        # ===================================================================
        # dataset = pd.read_csv("C:\\Backup\\DIS\\archive\\jacardAnomalies.csv")
        #
        # sns.set_theme(style="darkgrid")
        # sns.displot(data=dataset, x="jaccard_anomaly_score", bins=100, aspect=2.5, kde=True)
        # plt.show()
        #
        # height = dataset["jaccard_anomaly_score"].values
        # f = Fitter(height, distributions=get_common_distributions())
        # f.fit()
        # f.summary()
        # print(f.get_best(method="sumsquare_error"))
        # ===================================================================

        startYear = 2000
        endYear = 2019
        file_manager = MyFileManager()

        global_metadata_directory = "C:\\Backup\\DIS\\archive\\"
        if execution_configuration["JSONs"] == "1" or execution_configuration["Profiles"] == "1":
            if execution_configuration["Duration"] == "0":
                year = str(startYear) + "_" + str(endYear)
                year_directory_path = Path(global_metadata_directory + "/" + year)
                file_manager.directory_manipulation(year_directory_path)
                myGraphManager = MyGraphManager(year_directory_path,
                                                global_metadata_directory,
                                                global_metadata_file, 500, year, file_manager)
                if execution_configuration["JSONs"] == "1":
                    myGraphManager.manipulate_data()
                if execution_configuration["Profiles"] == "1":
                    myGraphManager.manipulate_profiles()
            else:
                year = str("_" + str(startYear) + "_" + str(endYear))
                aggregation_directory_path = Path(global_metadata_directory + "/" + year)
                file_manager.directory_manipulation(aggregation_directory_path)
                for year in range(startYear, endYear + 1):
                    year = str(year)
                    year_directory_path = Path(aggregation_directory_path.__str__() + "/" + year)
                    file_manager.directory_manipulation(year_directory_path)
                    myGraphManager = MyGraphManager(year_directory_path,
                                                    global_metadata_directory,
                                                    global_metadata_file, 500, year, file_manager)
                    if execution_configuration["JSONs"] == "1":
                        myGraphManager.manipulate_data()

        if execution_configuration["Results"] == "1":
            if execution_configuration["Duration"] == "0":
                year = str(startYear) + "_" + str(endYear)
                year_directory_path = Path(global_metadata_directory + "/" + str(year))
                myResultManager = MyResultManager(year_directory_path, True)
                myResultManager.manipulate_results(True, True, True)
            else:
                year = str("_" + str(startYear) + "_" + str(endYear))
                aggregation_directory_path = Path(global_metadata_directory + "/" + year)
                myResultManager = MyResultManager(aggregation_directory_path, True)
                myResultManager.update_time_results(startYear, endYear)
                myResultManager = MyResultManager(aggregation_directory_path, True)
                myResultManager.manipulate_time_results(startYear, endYear, sub_graph_nodes)

    except Exception as error:
        print("===================================================================================================")
        print(datetime.datetime.now().strftime("%D %H:%M:%S"))
        print("Error on_main: %s" % str(error))
        print("===================================================================================================")
        traceback.print_exc()
        print("===================================================================================================")






