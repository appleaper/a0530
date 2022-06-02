from util.tool import read_txt

def class_num(path):
    data = read_txt(path)
    new_dict = {}
    for line in data:
        line = line.replace('\n', '')
        path, label = line.split('\t')
        if label not in new_dict:
            new_dict[label] = 1
        else:
            new_dict[label] +=1
    print(new_dict)

if __name__ == '__main__':
    path = 'all.txt'
    class_num(path)