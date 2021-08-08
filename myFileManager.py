import shutil
import os
import csv


class MyFileManager:

    def __init__(self):
        self.path = None
        self.csv_newline = ""
        self.csv_encoding = "utf-8"

    def directory_manipulation(self, path):
        self.path = path
        if self.path.exists() and self.path.is_dir():
            shutil.rmtree(self.path)
        os.mkdir(self.path)

    def csv_write(self, csv_file, csv_columns, dict_data):
        with open(csv_file, "w", newline=self.csv_newline, encoding=self.csv_encoding) as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writeheader()
            for item in dict_data:
                writer.writerow(dict_data[item])




