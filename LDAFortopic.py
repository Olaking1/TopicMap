# -*- coding: utf-8 -*-
import gensim
from gensim import corpora, models, similarities
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
# import regex as re
import string
import os, re, time, logging
import jieba
import pickle as pkl


# logging.basicConfig(level=logging.WARNING,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',
#                     )

class loadFolders(object):  # 迭代器
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
                        # this_file = open(file_path, 'rb')
                        # content = this_file.read()
                        this_file.close()
                    yield catg, file, content
                    this_file.close()


def convert_doc_to_wordlist(str_doc, cut_all):
    sent_list = str(str_doc).split('\n')
    # sent_list = map(rm_char, sent_list)  # 去掉一些字符，例如中文空格
    jieba.load_userdict('../../datasets/use_words')
    word_2dlist = [rm_tokens(jieba.cut(re.sub("\p{P}+", "", part.lower()), cut_all=cut_all)) for part in
                   sent_list]  #分词
    word_list = sum(word_2dlist, [])
    return word_list

def rm_tokens(words):  # 去掉一些停用词和数字
    words_list = list(words)
    stop_words = get_stop_words()
    str = ''
    for i in range(len(string.punctuation)):
        str += ' '

    for i in range(words_list.__len__())[::-1]:
        table = words_list[i].maketrans(string.punctuation, str)
        words_list[i] = words_list[i].translate(table)
        if words_list[i] in stop_words:  # 去除停用词
            words_list.pop(i)
        elif words_list[i] in gensim.parsing.preprocessing.STOPWORDS:  # 去掉停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
        # elif words_list[i].isspace():
        #     words_list.pop(i)
        # words_list[i] = gensim.parsing.preprocessing.strip_punctuation(words_list[i])
    return words_list


def get_stop_words(path='../../datasets/stop_words'):
    file = open(path, 'r', encoding='utf-8')
    file = file.read().split('\n')
    return set(file)

# def rm_char(text):
#     text = re.sub(' ', '', text)
#     return text

if __name__ == '__main__':
    path_doc_root = '../../datasets/csdn_after'  # 根目录 即存放按类分类好的文本集
    path_tmp = '../../datasets/csdn_tmp'  # 存放中间结果的位置
    path_dictionary = os.path.join(path_tmp, 'THUNews.dict')
    path_tmp_corpus = os.path.join(path_tmp, 'corpus')
    path_tmp_tfidf = os.path.join(path_tmp, 'tfidf_corpus')
    path_tmp_lda = os.path.join(path_tmp, 'lda_corpus')
    path_tmp_ldamodel = os.path.join(path_tmp, 'lda_model.pkl')
    path_tmp_predictor = os.path.join(path_tmp, 'predictor.pkl')
    n = 1  # n 表示抽样率， n抽1
    n_topic = 9

    dictionary = None
    corpus = []
    filenames = []
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
    if not os.path.exists(path_dictionary):
        print('=== 未检测到有词典存在，开始遍历生成词典 ===')
        dictionary = corpora.Dictionary()
        files = loadFiles(path_doc_root)
        for i, msg in enumerate(files):
            if i % n == 0:
                catg = msg[0]
                filename = msg[1]
                file = msg[2]
                file = convert_doc_to_wordlist(file, cut_all=False)
                dictionary.add_documents([file])
                filenames.append(filename)
                if int(i / n) % 1000 == 0:
                    print('{t} *** {i} \t docs has been dealed'
                          .format(i=i, t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        print("去掉出现次数过多或过少的词前，字典长度为：" + str(len(dictionary)))
        # 去掉词典中出现次数过少或过多的
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 20]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()
        dictionary.save(path_dictionary)
        print("去掉出现次数过多或过少的词后，字典长度为：" + str(len(dictionary)))
        print('=== 词典已经生成 ===')
    else:
        print('=== 检测到词典已经存在，跳过该阶段 ===')

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
        lda_model = models.LdaModel(corpus=corpus_tfidf_total, id2word=dictionary, num_topics=n_topic, alpha='auto',
                                    eval_every=1)
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

    # # ===================================================================
    # # # # 第四阶段，  计算文档之间的相似度
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

    doc = []
    if(len(filenames) == 0):
        files = loadFiles(path_doc_root)
        for i, msg in enumerate(files):
            if i % n == 0:
                filename = msg[1]
                filenames.append(filename)
                doc.append(msg[2])

    gamma = lda_model.do_estep(corpus, state=lda_model.state)
    lda_model.update_alpha(gamma, 0.7)
    ldaState = models.ldamodel.LdaState(eta=0.7, shape=(lda_model.num_topics, lda_model.num_terms))
    lda_model.optimize_eta = True
    lda_model.do_mstep(rho=0.3, other=ldaState)
    topic = lda_model.show_topics(n_topic, 5)
    print(topic)
    # index = similarities.docsim.Similarity(output_prefix=path_tmp, corpus=corpus,
    #                                        num_features=len(dictionary))
    # filePath = os.path.join(path_tmp,'similaritiesFiles')
    # file = open(filePath, 'a', encoding='utf-8')
    # for i, similarities in zip(range(len(dictionary)), index):
    #     file.writelines("================================================")
    #     file.write('\r\n')
    #     file.writelines("【【【%d 与 %s相似的文档为：】】】"%(i, filenames[i]))
    #     file.write('\r\n')
    #     boolSimi = similarities > 0.85
    #     for j in range(len(boolSimi)):
    #         if(boolSimi[j]):
    #             file.writelines(str(similarities[j])+":"+filenames[j])
    #             file.write('\r\n')
    # file.close()

    documentTopic = lda_model.get_document_topics(corpus[2],minimum_phi_value=0.02)
    topicList = lda_model.get_topic_terms(1,10)
    for i in range(len(documentTopic)):
        print(dictionary[documentTopic[i][0]])

    print("Log perplexity of the model is", lda_model.log_perplexity(corpus))