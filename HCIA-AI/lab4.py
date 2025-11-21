import pandas as pd
from sklearn import tree
import pydotplus

"""生成决策树"""


def create_tree(training_data):
    datas = training_data.iloc[:, :-1]  # 特征矩阵
    labels = training_data.iloc[:, -1]  # 标签
    trained_tree = tree.DecisionTreeClassifier(criterion="entropy")  # 分类决策树
    trained_tree.fit(datas, labels)
    return trained_tree


def show_tree2pdf(trained_tree, file_name):
    dot_data = tree.export_graphviz(trained_tree, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(file_name)  # 保存为pdf文件


def data2vector(data):
    names = data.columns[:-1]
    for i in names:
        codes = pd.Categorical(data[i]).codes
        data[i] = codes
    return data


def main():
    data = pd.read_csv("../dataset/ML/tennis.txt", header=None, sep="\t")
    train = data2vector(data)
    decision_tree = create_tree(train)
    show_tree2pdf(decision_tree, "tennis.pdf")
    #testVec = [0,0,1,1] # 天气晴、气温冷、湿度高、风力强
    #print(decision_tree.predict(np.array(testVec).reshape(1,-1))) #预测

if __name__ == '__main__':
    main()
