import re


class Mos:
    __slots__=('name','type', 'left', 'mid', 'right', 'w', 'l',)
    def __init__(self, name,left, mid, right, type, w, l):
        self.name=name
        self.left = left
        self.mid = mid
        self.right = right
        self.type = type
        self.w = w
        self.l = l
class Parser:
    def __init__(self) -> None:
        self.path = None
        self.mos_list = []
        self.attributes = ('name','type', 'left', 'mid', 'right', 'w', 'l')
        self.words_list=[]
        self.pattern = re.compile(r"(.*\s)(.*\s)(.*\s)(.*\s)(.*\s).*\sW=(.*)u\sL=(.*)n\n")
    def parse(self,path):
        self.path = path
        with open(self.path, 'r') as f:
            for line in f:
                line_data = re.match(self.pattern, line)
                if line_data:
                    name, left, mid, right, type, w, l = line_data.groups()
                    print(name, left, mid, right, type, float(w)*1000, float(l))
                    self.mos_list.append(Mos(name, left, mid, right,
                            type, float(w)*1000, float(l)))
                    self.words_list.append([name, left, mid, right, type, float(w)*1000, float(l)])
        return self.mos_list,self.words_list


if __name__ == "__main__":
    paser = Parser()
    paser.pattern=re.compile(r"(.*\s)(.*\s)(.*\s)(.*\s)(.*\s).*\sW=(.*)u\sL=(.*)n\n")
    mos_list,words_list = paser.parse('test1.nets')