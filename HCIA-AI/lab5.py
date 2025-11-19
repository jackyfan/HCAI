
from numpy import *
from os import listdir
import codecs #字符转换模块，用于文本的编码和解码
import jieba #中文分词库
import re
from sklearn.naive_bayes import  MultinomialNB
from collections import Counter
from itertools import chain #用于串联迭代对象

"""
通过 jieba 文字分词库对邮件数据集的垃圾邮件和进行文本处理，提取特征。
然后调用 sklearn机器学习库中的朴素贝叶斯算法训练模型，最后推理测试集中邮件是否为垃圾邮件。
"""
def segment2word(doc:str):
    #从stop_list.txt文件提取停用词
    stop_words = codecs.open("../dataset/ML/04/stop_list.txt",'r','UTF-8').read().splitlines()
    doc = re.sub('[\t\r\n]',' ',doc)#去掉邮件文本中的换行符、制表符、回车符
    word_list = list(jieba.cut(doc.split())) #用jieba分词
    out_str=''
    for word in word_list:
        if word=='' or word==' ':
             continue
        if word not in stop_words:
            out_str += word.strip()
            out_str += ' '
    segments = out_str.strip().split(sep=' ')
    return segments

def get_data_from_dir(data_dir):
    docs = []
    labels = [f for f in listdir(data_dir) if f.endswith('.txt')]
    for doc in labels:
        try:
            docpath = data_dir + '/' + doc
            words = segment2word(codecs.open(docpath,'r','UTF-8').read())
            docs.append(words)
        except:
            print("handling file %s is error"%docpath)
    return docs

