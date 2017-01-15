# -*- coding: utf-8 -*-
import gensim
from gensim import corpora, models, similarities
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import numpy as np
import string
import os, re, time, logging
import jieba
import jieba.posseg
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt

# logging.basicConfig(level=logging.WARNING,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',
#                     )


class loadFolders(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath):  # if file is a folder
                yield file_abspath


class loadFiles(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:  # level directory
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):  # secondary directory
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    this_file = open(file_path, 'r', encoding='utf-8')
                    try:
                        content = this_file.read()
                    except UnicodeDecodeError:
                        print(file_path)
                        content = ''
                    yield catg, file, content
                    this_file.close()


def convert_doc_to_wordlist(str_doc, cut_all):
    sent_list = str(str_doc).split('\n')
    # sent_list = map(rm_char, sent_list)  # 去掉一些字符，例如中文空格
    jieba.load_userdict('use_words')
    word_2dlist = []
    for part in sent_list:
        list_all = jieba.posseg.cut(part.lower())
        list_numn = []
        for i in list_all:
            if(i.flag is 'n' or i.flag is 'eng' or i.flag is 'x'):
                list_numn.append(i.word)
        word_2dlist.append(rm_tokens(list_numn))
    word_list = sum(word_2dlist, [])
    return word_list


def rm_tokens(words):  # 去掉一些停用词和数字
    words_list = list(words)
    stop_words = get_stop_words()
    str = ' '*len(string.punctuation)

    for i in range(words_list.__len__())[::-1]:
        table = words_list[i].maketrans(string.punctuation, str)
        words_list[i] = words_list[i].translate(table)
        if words_list[i] in stop_words:  # 去除停用词
            words_list.pop(i)
        elif words_list[i] in gensim.parsing.preprocessing.STOPWORDS:  # 去掉停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list


def get_stop_words(path='stop_words'):
    file = open(path, 'r', encoding='utf-8')
    file = file.read().split('\n')
    return set(file)

def accuracy(filenames, fileTopicList): # 计算分类的正确性
    count = 0
    # docfre＝40 accuracy=57.21%
    # docsTopic = ['linux','hbase','mysql','docker','mongodb','zookeeper','mysql','hadoop']
    # docfre ＝ 50 accrucy＝59.24%
    docsTopic = ['mybatis','linux','mongodb','docker','zookeeper','mysql','hbase','hadoop']
    # docfre = 60 accuray=60.77%
    # docsTopic = ['mybatis','docker','mongodb','hbase','mysql','hadoop','linux','zookeeper']
    # docfre = 70 accuray=66.81%
    # docsTopic = ['mongodb','zookeeper','hadoop','linux','hbase','docker','mybatis','mysql']
    # docfre = 80 accuray=52.47%
    docsTopic = ['mybatis', 'linux', 'docker', 'mysql', 'zookeeper', 'hadoop', 'hbase', 'mongodb']
    # docfre = 90 accuray=52.47%
    docsTopic = ['zookeeper','linux','mongodb','hbase','docker','zookeeper','hadoop','mysql']
    for i in range(len(filenames)):
        try:
            if filenames[i].lower().__contains__(docsTopic[fileTopicList[i][0]]):
                count = count+1
        except Exception:
            print(i)
    return count/len(filenames)


if __name__ == '__main__':
    path_doc_root = '../datasets/csdn_after'  # 根目录 即存放按类分类好的文本集
    path_tmp = '../datasets/csdn_tmp'  # 存放中间结果的位置
    path_dictionary = os.path.join(path_tmp, 'THUNews.dict')
    path_tmp_corpus = os.path.join(path_tmp, 'corpus')
    path_tmp_tfidf = os.path.join(path_tmp, 'tfidf_corpus')
    path_tmp_lda = os.path.join(path_tmp, 'lda_corpus')
    path_tmp_ldamodel = os.path.join(path_tmp, 'lda_model.pkl')
    path_tmp_predictor = os.path.join(path_tmp, 'predictor.pkl')
    path_temp_filenames = os.path.join(path_tmp, 'filenames')
    n = 5  # n 表示抽样率， n抽1
    n_topic = 8

    dictionary = None
    corpus = []
    filenames = []
    corpus_topic = []
    corpus_tfidf = None
    corpus_lda = None
    lda_model = None
    predictor = None
    word_2dlist = None
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)
    # # ===================================================================
    # # # # 第一阶段，  遍历文档，生成词典,并去掉频率较少的项
    #       如果指定的位置没有词典，则重新生成一个。如果有，则跳过该阶段
    t0 = time.time()
    if not os.path.exists(path_dictionary):
        print('=== 未检测到有词典存在，开始遍历生成词典 ===')
        dictionary = corpora.Dictionary()
        files = loadFiles(path_doc_root)
        filenames_file = open(path_temp_filenames, 'w', encoding='utf-8')
        for i, msg in enumerate(files):
            if i % n == 0:
                catg = msg[0]
                filename = msg[1]
                file = msg[2]
                file = file + filename*5
                word_list = convert_doc_to_wordlist(file, cut_all=False)
                dictionary.add_documents([word_list])
                filenames.append(filename)
                # 将所有文件名存储到一个文件中，方便以后使用
                filenames_file.writelines(filename+'\n')
                if int(i / n) % 1000 == 0:
                    print('{t} *** {i} \t docs has  been dealed'
                          .format(i=i, t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        filenames_file.close()

        print("去掉出现次数过多或过少的词前，字典长度为：" + str(len(dictionary)))
        # 去掉词典中出现次数过少或过多的
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 80]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()
        dictionary.save(path_dictionary)
        print("去掉出现次数过多或过少的词后，字典长度为：" + str(len(dictionary)))
        print('=== 词典已经生成 ===')
    else:
        print('=== 检测到词典已经存在，跳过该阶段 ===')
    t1 = time.time()
    print("第一阶段用时：%d" % (t1-t0))

    # # ===================================================================
    # # # # 第二阶段，  开始将文档转化成tfidf
    if not os.path.exists(path_tmp_tfidf):
        print('=== 未检测到有tfidf文件夹存在，开始生成tfidf向量 ===')
        # 如果指定的位置没有tfidf文档，则生成一个。如果有，则跳过该阶段
        if not dictionary:  # 如果跳过了第一阶段，则从指定位置读取词典
            dictionary = corpora.Dictionary.load(path_dictionary)
        os.makedirs(path_tmp_tfidf)
        files = loadFiles(path_doc_root)
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        corpus_tfidf = {}
        for i, msg in enumerate(files):
            if i % n == 0:
                catg = msg[0]
                file = msg[1]
                word_list = convert_doc_to_wordlist(file, cut_all=False)
                file_bow = dictionary.doc2bow(word_list)
                corpus.append(file_bow)
                file_tfidf = tfidf_model[file_bow]
                tmp = corpus_tfidf.get(catg, [])
                tmp.append(file_tfidf)
                if tmp.__len__() == 1:
                    corpus_tfidf[catg] = tmp
            if i % 10000 == 0:
                print('{i} files is dealed'.format(i=i))
        # 将corpus中间结果存储起来
        os.mkdir(path_tmp_corpus)
        corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_corpus, s=os.sep, c='corpus'), corpus,
                                   id2word=dictionary)
        # 将tfidf中间结果储存起来
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg),
                                       corpus_tfidf.get(catg),
                                       id2word=dictionary
                                       )
            print('catg {c} has been transformed into tfidf vector'.format(c=catg))
        print('=== tfidf向量已经生成 ===')
    else:
        print('=== 检测到tfidf向量已经生成，跳过该阶段 ===')
    t2 = time.time()
    print("第二阶段用时：%d" % (t2-t1))

    # # ===================================================================
    # # # # 第三阶段，  开始将tfidf转化成lda
    if not os.path.exists(path_tmp_lda):
        print('=== 未检测到有lda文件夹存在，开始生成lda向量 ===')
        if not dictionary:
            dictionary = corpora.Dictionary.load(path_dictionary)
        if not corpus_tfidf:  # 如果跳过了第二阶段，则从指定位置读取tfidf文档
            print('--- 未检测到tfidf文档，开始从磁盘中读取 ---')
            # 从对应文件夹中读取所有类别
            files = os.listdir(path_tmp_tfidf)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)

            # 从磁盘中读取corpus
            corpus_tfidf = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg)
                tfidfCorpus = corpora.MmCorpus(path)
                corpus_tfidf[catg] = tfidfCorpus
            print('--- tfidf文档读取完毕，开始转化成lda向量 ---')

        # 生成lda model
        os.makedirs(path_tmp_lda)
        corpus_tfidf_total = []
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            tmp = corpus_tfidf.get(catg)
            corpus_tfidf_total += tmp
        lda_model = models.LdaModel(corpus=corpus_tfidf_total, id2word=dictionary, num_topics=n_topic, alpha=0.1,
                                    eval_every=1, iterations=100)

        # 将lda模型存储到磁盘上
        lda_file = open(path_tmp_ldamodel, 'wb')
        pkl.dump(lda_model, lda_file)
        lda_file.close()
        del corpus_tfidf_total  # lda model已经生成，释放变量空间
        print('--- lda模型已经生成 ---')

        # 生成corpus of lda, 并逐步去掉 corpus of tfidf
        corpus_lda = {}
        for catg in catgs:
            corpu = [lda_model[doc] for doc in corpus_tfidf.get(catg)]
            corpus_lda[catg] = corpu
            corpus_tfidf.pop(catg)
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_lda, s=os.sep, c=catg),
                                       corpu,
                                       id2word=dictionary)
        print('=== lda向量已经生成 ===')
    else:
        print('=== 检测到lda向量已经生成，跳过该阶段 ===')
    t3 = time.time()
    print("第三阶段用时：%d" % (t3-t2))

    # # ===================================================================
    # # # # 第四阶段，  计算文档之间的相似度，并画图表示
    if not corpus_lda:  # 如果跳过了第三阶段
        print('--- 未检测到lda文档，开始从磁盘中读取 ---')
        files = os.listdir(path_tmp_lda)
        catg_list = []
        for file in files:
            t = file.split('.')[0]
            if t not in catg_list:
                catg_list.append(t)
        # 从磁盘中读取corpus
        path = '{f}{s}{c}.mm'.format(f=path_tmp_corpus, s=os.sep, c='corpus')
        corpus = corpora.MmCorpus(path)
        # 从磁盘中读取corpus_lda
        corpus_lda = {}
        for catg in catg_list:
            path = '{f}{s}{c}.mm'.format(f=path_tmp_lda, s=os.sep, c=catg)
            Mmcorpus = corpora.MmCorpus(path)
            corpus_lda[catg] = Mmcorpus
        # 从磁盘中读取dictionary
        dictionary = corpora.Dictionary.load(path_dictionary)
        # 从磁盘中读取lda_model
        lda_file = open(path_tmp_ldamodel, "rb")
        lda_model = pkl.load(lda_file, encoding="utf-8")
        print('--- lda文档读取完毕，开始进行分类 ---')
    t4 = time.time()
    print("第四阶段用时：%d" % (t4-t3))

    gamma = lda_model.do_estep(corpus)
    lda_model.update_alpha(gamma, 0.7)
    ldaState = models.ldamodel.LdaState(eta=0.3, shape=(lda_model.num_topics, lda_model.num_terms))
    # lda_model.optimize_eta = True
    lda_model.do_mstep(rho=0.7, other=ldaState)

    topic = lda_model.show_topics(n_topic, 5)
    print(topic)

    # # ===================================================================
    # # # # 第五阶段，  计算文档主题的准确率
    # 从文件中读取filenames并转化为列表形式
    filenames = []
    f = open(path_temp_filenames, 'r', encoding='utf-8')
    for line in f:
        filenames.append(line.strip('\n'))
    f.close()

    fileTopicList = []
    for j in range(len(filenames)):
        fileTopic = lda_model.get_document_topics(corpus[j], minimum_phi_value=0.02)
        topicList = lda_model.get_topic_terms(1,10)
        fileTopic.sort(key=lambda x:x[1], reverse=True)
        fileTopicList.append(fileTopic[0])

    accuracy = accuracy(filenames, fileTopicList)
    print(accuracy)
    print("Log perplexity of the model is", lda_model.log_perplexity(corpus))

    # ff = open('similarity','w',encoding='utf-8')
    graph = nx.Graph()
    doc_index = similarities.docsim.Similarity(output_prefix=path_tmp, corpus=corpus, num_features=len(dictionary))
    # doc_index = similarities.MatrixSimilarity(lda_model[corpus])
    for i, similars in zip(range(len(filenames)), doc_index):
        fileTopicI = lda_model.get_document_topics(corpus[i], minimum_phi_value=0.02)
        fileTopicI.sort(key=lambda x: x[1], reverse=True)
        # ff.write("与<<"+filenames[i]+">>相似的文档有：")
        # ff.write('\n')
        for j in range(len(similars)):
            if(similars[j] >0.75 and similars[j] < 0.99):
                # ff.write(filenames[j]+"相似度为"+str(similarities[j]))
                # ff.write('\n')
                graph.add_node(filenames[i], topic=fileTopicI[0][0])
                graph.add_edge(filenames[i],filenames[j], weight=similars[j])
        # ff.write("============================")
        # ff.write("\n")
    # ff.close()

    # # ===================================================================
    # # # # 第五阶段，  计算主题之间的相似度
    topn = 300
    topic_term = []
    for i in range(n_topic):
        topic_term_pro = lda_model.get_topic_terms(i, topn=topn)
        topic_term = [(id,pro) for (id, pro) in topic_term_pro]
        corpus_topic.append(topic_term)
    topic_index = similarities.docsim.Similarity(output_prefix=path_tmp, corpus=corpus_topic, num_features=len(dictionary))
    for i, t_similars in zip(range(n_topic), topic_index):
        for j in range(len(t_similars)):
            if(t_similars[j]>0.5 and t_similars[j]<1.0):
                print(str(i)+"--"+str(j)+"  "+str(t_similars[j]))

    pos = nx.spring_layout(graph)
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] >=0.90]
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <0.90]
    nodeColor = [d['topic'] for (n, d) in graph.nodes(data=True)]
    nx.draw_networkx_nodes(graph, pos, node_color=nodeColor, node_size=100)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=1, with_labels=False)
    nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=1, style='dashed', with_labels=False)
    plt.show()