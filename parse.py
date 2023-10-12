import re


class Mos:
    def __init__(self, name, type, left, mid, right, w, l):
        self.index = 0
        pass


def parse(path):
    """解析晶体管文件"""
    mos_list = []
    pattern = re.compile(
        r"(.*\s)(.*\s)(.*\s)(.*\s)(.*\s).*\sW=(.*)u\sL=(.*)n\n")
    with open(path, 'r') as f:
        for line in f:
            line_data = re.match(pattern, line)
            if line_data:
                name, left, mid, right, type, w, l = line_data.groups()
                print(name, left, mid, right, type, float(w)*1000, float(l))
                mos_list.append(Mos(name, left, mid, right,
                                type, float(w)*1000, float(l)))
    return mos_list


if __name__ == "__main__":
    mos_list = parse('test1.nets')