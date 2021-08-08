import traceback
import datetime
from pathlib import Path
from myGraphManager import MyGraphManager
from myFileManager import MyFileManager
from myResultManager import MyResultManager

if __name__ == "__main__":
    execution_sequence = "0010"
    global_metadata_file = "metadata.csv"

    try:
        startYear = 2015
        endYear = 2015
        execution_sequence_array = [char for char in execution_sequence]

        global_metadata_directory = "C:\\Users\\Msi-PC\\OneDrive - Impedimed\\DIS\\archive\\"
        if execution_sequence_array[0] == "1" or execution_sequence_array[1] == "1":
            for year in range(startYear, endYear + 1):
                year_directory_path = Path(global_metadata_directory + "/" + str(year))
                file_manager = MyFileManager()
                file_manager.directory_manipulation(year_directory_path)
                myGraphManager = MyGraphManager(year_directory_path,
                                                global_metadata_directory,
                                                global_metadata_file, 500, year, file_manager)
                if execution_sequence_array[0] == "1":
                    myGraphManager.manipulate_data()
                if execution_sequence_array[1] == "1":
                    myGraphManager.manipulate_profiles()

        # global_metadata_directory = "C:\\Users\\Msi-PC\\Desktop\\DIS\\"
        if execution_sequence_array[2] == "1":
            if execution_sequence_array[3] == "0":
                for year in range(startYear, endYear + 1):
                    year_directory_path = Path(global_metadata_directory + "/" + str(year))
                    myResultManager = MyResultManager(year_directory_path, True, None)
                    myResultManager.manipulate_results(False, False, False, [-4528402695832970487,
                                                                             -6427164912164990441,
                                                                             6694179879672183187,
                                                                             -4120636499414657878,
                                                                             -3163456006207561760])
            else:
                years = "2000_2020"
                year_directory_path = Path(global_metadata_directory + "/" + str(years))
                myResultManager = MyResultManager(year_directory_path, True, global_metadata_directory)
                myResultManager.manipulate_results(False, False, False, [-4528402695832970487,
                                                                        -6427164912164990441,
                                                                        6694179879672183187,
                                                                        -4120636499414657878,
                                                                        -3163456006207561760])
    except Exception as error:
        print("===================================================================================================")
        print(datetime.datetime.now().strftime("%D %H:%M:%S"))
        print("Error on_main: %s" % str(error))
        print("===================================================================================================")
        traceback.print_exc()
        print("===================================================================================================")

