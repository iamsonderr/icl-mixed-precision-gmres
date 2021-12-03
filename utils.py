import csv

class open_history_file(object):
    def __init__(self, mat):
        self.mat = mat

    def __enter__(self):
        self.file = open('history-'+self.mat+'.csv', 'r')
        try:
            self.reader = csv.DictReader(self.file, delimiter=',', fieldnames=['mat', 'type', 'orth', 'rlen', 'rtol', 'rorth', 'tol', 'device', 'prec', 'i', 'total_iters', 'res', 'err', 'ilu', 'gmres'])
            return self.reader
        except err:
            self.file.close()
            raise
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


def process_rows(mat, baseline_func, mixed_func, singleprec_func, single_func,
                 tol=None, orth=None, device=None, prec=None, rlen=None, rtol=None, rorth=None):
    with open_history_file(mat) as reader:
        for row in reader:
            if (row and (tol == None or tol == row['tol'])
                    and (orth == None or orth == row['orth'])
                    and (device == None or device == row['device'])
                    and (prec == None or prec == row['prec'])
                    and (rlen == None or rlen == row['rlen'])
                    and (rtol == None or rtol == row['rtol'])
                    and (rorth == None or rorth == row['rorth'])):
                if row['type'] == 'b':
                    baseline_func(row)
                elif row['type'] == 'mp':
                    mixed_func(row)
                elif row['type'] == 'p':
                    singleprec_func(row)
                elif row['type'] == 's':
                    single_func(row)
