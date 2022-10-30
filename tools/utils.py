import re
import errno
import os
import pandas as pd


def is_integer(number):
    number = re.sub(r"(?<=\d)[\,](?=\d)", "", number)  # "12,456" -> "12456"
    if re.findall(r"^-?[\d]+$", number):
        return True
    else:
        return False


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def convert_table_text_to_pandas(table_text):
    _data = {}

    table_text = re.sub(r" ?\| ?", " | ", table_text)
    cells = [row.split(" | ") for row in table_text.split("\n")]

    row_num = len(cells)
    column_num = len(cells[0])

    # for table without a header
    first_row = cells[0]
    matches = re.findall(r"[\d]+", " ".join(first_row))
    if len(matches) > 0:
        header = [f"Column {i+1}" for i in range(column_num)]
        cells.insert(0, header)

    # build DataFrame for the table
    for i in range(column_num):
        _data[cells[0][i]] = [row[i] for row in cells[1:]]

    table_pd = pd.DataFrame.from_dict(_data)

    return table_pd


class Logger(object):

    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + '\n')
        self.log_file.flush()
        print(msg)
