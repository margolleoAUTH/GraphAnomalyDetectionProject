import shutil
import os
import stat
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
        os.chmod(self.path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    def dict_csv_write(self, csv_file, csv_columns, dict_data):
        with open(csv_file, "w", newline=self.csv_newline, encoding=self.csv_encoding) as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writeheader()
            for item in dict_data:
                writer.writerow(dict_data[item])

    def df_csv_write(self, csv_file, df):
        df.to_csv(csv_file, index=False, header=True, encoding=self.csv_encoding)


