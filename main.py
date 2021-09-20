import traceback
import datetime
from pathlib import Path
from myGraphManager import MyGraphManager
from myFileManager import MyFileManager
from myResultManager import MyResultManager

if __name__ == "__main__":
    execution_configuration = {
        "JSONs": "1",
        "Profiles": "1",
        "Results": "1",
        "Duration": "1"
    }
    global_metadata_file = "metadata.csv"
    sub_graph_nodes = []
    try:
        startYear = 2000
        endYear = 2019
        file_manager = MyFileManager()

        global_metadata_directory = "C:\\Users\\Msi-PC\\OneDrive - Impedimed\\DIS\\archive\\"
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
                myResultManager.manipulate_results(True, True, True, sub_graph_nodes)
            else:
                year = str("_" + str(startYear) + "_" + str(endYear))
                aggregation_directory_path = Path(global_metadata_directory + "/" + year)
                myResultManager = MyResultManager(aggregation_directory_path, True)
                myResultManager.update_time_results(startYear, endYear)
                myResultManager = MyResultManager(aggregation_directory_path, True)
                myResultManager.manipulate_time_results(startYear, endYear)

    except Exception as error:
        print("===================================================================================================")
        print(datetime.datetime.now().strftime("%D %H:%M:%S"))
        print("Error on_main: %s" % str(error))
        print("===================================================================================================")
        traceback.print_exc()
        print("===================================================================================================")






