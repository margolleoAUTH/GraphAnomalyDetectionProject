import traceback
from myGraphManager import MyGraphManager
import datetime

if __name__ == "__main__":
    execution_sequence = "001"
    pure_init_file = "metadata.csv"

    try:
        directory = "C:\\Users\\user\\Desktop\\DIS\\archive\\"
        myGraphManager = MyGraphManager(directory, pure_init_file, 2015, 500, True)
        execution_sequence_array = [char for char in execution_sequence]

        if execution_sequence_array[0] == "1":
            myGraphManager.manipulate_data()
        if execution_sequence_array[1] == "1":
            myGraphManager.manipulate_profiles()
        if execution_sequence_array[2] == "1":
            myGraphManager.manipulate_results([], True, False, False)
    except Exception as error:
        print("===================================================================================================")
        print(datetime.datetime.now().strftime("%D %H:%M:%S"))
        print("Error on_main: %s" % str(error))
        print("===================================================================================================")
        traceback.print_exc()
        print("===================================================================================================")

