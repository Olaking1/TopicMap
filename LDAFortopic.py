# -*- coding: utf-8 -*-
import gensim
from gensim import corpora, models, similarities
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import numpy as np
import scipy
import string
import os, re, time, logging
import jieba
import jieba.posseg
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

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
    # docfre = 70 accuray=66.81%
    docsTopic = ['mongodb','zookeeper','hadoop','linux','hbase','docker','mybatis','mysql']
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
    path_temp_topicmaps = os.path.join(path_tmp, 'topic_map')
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
                content = msg[2]
                content = content + filename * 5
                word_list = convert_doc_to_wordlist(content, cut_all=False)
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
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 30]
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
                content = msg[1]
                word_list = convert_doc_to_wordlist(content, cut_all=False)
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
            for content in files:
                t = content.split('.')[0]
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
    # # # # 第四阶段，   读取存在本地的lda模型
    if not corpus_lda:  # 如果跳过了第三阶段
        print('--- 未检测到lda文档，开始从磁盘中读取 ---')
        files = os.listdir(path_tmp_lda)
        catg_list = []
        for content in files:
            t = content.split('.')[0]
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
    # # # # 第五阶段，  计算主题之间的相似度
    topn = 300
    topic_term = []
    for i in range(n_topic):
        topic_term_pro = lda_model.get_topic_terms(i, topn=topn)
        topic_term = [(id, pro) for (id, pro) in topic_term_pro]
        corpus_topic.append(topic_term)
    topic_index = similarities.docsim.Similarity(output_prefix=path_tmp, corpus=corpus_topic,
                                                 num_features=len(dictionary))
    for i, t_similars in zip(range(n_topic), topic_index):
        for j in range(len(t_similars)):
            if (t_similars[j] > 0.5 and t_similars[j] < 1.0):
                print(str(i) + "--" + str(j) + "  " + str(t_similars[j]))

    # # ===================================================================
    # # # # 第六阶段，  计算文档主题的准确率
    # 从文件中读取filenames并转化为列表形式
    filenames = []
    f = open(path_temp_filenames, 'r', encoding='utf-8')
    for line in f:
        filenames.append(line.strip('\n'))
    f.close()

    # fileTopicList = []
    # for j in range(len(filenames)):
    #     fileTopic = lda_model.get_document_topics(corpus[j], minimum_phi_value=0.02)
    #     topicList = lda_model.get_topic_terms(1,10)
    #     fileTopic.sort(key=lambda x:x[1], reverse=True)
    #     fileTopicList.append(fileTopic[0])
    #
    # accuracy = accuracy(filenames, fileTopicList)
    # print(accuracy)

    # # ===================================================================
    # # # # 第七阶段，  画出主题地图，并计算各网络的相关参数
    if not os.path.exists(path_temp_topicmaps):
        graph = nx.Graph()
        doc_index = similarities.docsim.Similarity(output_prefix=path_tmp, corpus=corpus, num_features=len(dictionary))
        for i, similars in zip(range(len(filenames)), doc_index):
            fileTopicI = lda_model.get_document_topics(corpus[i], minimum_phi_value=0.02)
            fileTopicI.sort(key=lambda x: x[1], reverse=True)
            for j in range(len(similars)):
                if(similars[j] >0.70 and similars[j] < 0.99):
                    graph.add_node(i, topic=fileTopicI[0][0])
                    graph.add_edge(i,j, weight=similars[j])

        nx.write_gml(graph, path_temp_topicmaps, stringizer=str)
    else:
        graph = nx.read_gml(path_temp_topicmaps, destringizer=float)

    topic_map_cluster = nx.average_clustering(graph)
    print("平均路径长度为：%f" % nx.average_shortest_path_length(graph))

    degree_dict = nx.degree(graph)
    degree_dict = sorted(degree_dict.items(), key=lambda d:d[1], reverse=True)
    degree_distr = {}
    higher_degree_nodes = []
    count = 0
    for key, value in degree_dict:
        count = count + value
        if value in degree_distr:
            degree_distr[value] +=1
        else:
            degree_distr[value] = 1
        if value>=60:
            higher_degree_nodes.append(key)
    average_degree = count / len(nx.nodes(graph))

    print("主题地图平均度数：%f" % average_degree)
    print("主题地图的结点总数:%d" % len(nx.nodes(graph)))
    topicmap_path = nx.average_shortest_path_length(graph)

    regular_graph = nx.random_graphs.random_regular_graph(int(average_degree), len(nx.nodes(graph)))
    regular_graph_cluster = nx.average_clustering(regular_graph)
    regular_graph_path = nx.average_shortest_path_length(regular_graph)

    ER_graph = nx.random_graphs.erdos_renyi_graph(len(nx.nodes(graph)), 0.0079)
    ER_graph_cluster = nx.average_clustering(ER_graph)
    ER_graph_path = nx.average_shortest_path_length(ER_graph)

    WS_graph = nx.random_graphs.watts_strogatz_graph(len(nx.nodes(graph)), int(average_degree)*2, 0.0158)
    WS_graph_cluster = nx.average_clustering(WS_graph)
    WS_graph_path = nx.average_shortest_path_length(WS_graph)

    width = 1
    ind = np.linspace(1, 9, 4)
    Y1 = [regular_graph_cluster,ER_graph_cluster,WS_graph_cluster,topic_map_cluster]
    Y2 = [regular_graph_path, ER_graph_path, WS_graph_path, topicmap_path]
    labels = ['regular','random','small world','topic map']
    fig1 = plt.figure(1, dpi=250)
    ax1 = fig1.add_subplot(121)
    ax1.bar(ind-width/2,Y1,width)
    ax1.set_xticks(ind)
    ax1.set_ylabel('Clustering coefficient')
    ax1.set_xticklabels(labels, fontsize='medium', rotation=15)
    ax2 = fig1.add_subplot(122)
    ax2.bar(ind - width / 2, Y2, width)
    ax2.set_xticks(ind)
    ax2.set_ylabel('average shortest path length')
    ax2.set_ylim(0,10)
    ax2.set_xticklabels(labels, fontsize='medium', rotation=15)
    plt.subplots_adjust(left=0.08, bottom=0.30, right=0.98, top=0.70)
    fig1.savefig('compare with networkx.eps')
    fig1.savefig('compare with networkx.jpeg')

    fig2 = plt.figure(2, dpi=250)
    ax3 = fig2.add_subplot(111)
    X3 = np.array(list(degree_distr.keys()))
    Y3 = list(degree_distr.values())
    Y3_2 = 250*scipy.power(X3, -1.5)
    ax3.scatter(X3, Y3, marker='o', label='Degree distribution')
    ax3.set_title('Degree distributions of the topic map')
    ax3.plot(X3, Y3_2, linewidth=3, color='r', markeredgewidth=0.5, label='$P(k)=\\alpha*k^{-\\beta}$')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-20, 100)
    plt.legend()
    fig2.savefig('degree distribution.eps')
    fig2.savefig('degree distribution.jpeg')

    fig3 = plt.figure(3)
    ax4 = fig3.add_subplot(111)
    pos = nx.spring_layout(graph, iterations=600)
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] >=0.90]
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <0.90]
    nodeColor = [d['topic'] for (n, d) in graph.nodes(data=True)]
    nx.draw_networkx_nodes(graph, pos, node_color=nodeColor, node_size=300)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=1)
    nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=1)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')
    higher_degree_graph = nx.Graph()
    higher_degree_graph.add_nodes_from(higher_degree_nodes)
    nx.draw_networkx_nodes(higher_degree_graph, pos, node_shape='s', node_size=250, node_color='g')
    nx.draw_networkx_labels(higher_degree_graph, pos, font_size=10, font_family='sans-serif')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['right'].set_color('none')
    ax4.spines['top'].set_color('none')
    ax4.spines['bottom'].set_color('none')
    ax4.spines['left'].set_color('none')
    plt.title('Topic map of eight topic')
    fig3.savefig('topic map.eps')
    fig3.savefig('topic map.jpeg')
    plt.show()