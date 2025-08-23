import csv, os
def safe_mean(xs):
    return sum(xs)/max(1,len(xs))

class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = None

    def log(self, row: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
