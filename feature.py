from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from crawler import CacheNode
from textblob import TextBlob
from collections import Counter
import os
from sklearn.preprocessing import normalize
import cPickle as pickle
from lxml import html

class TreeNode:
    def __init__(self, cachePath):

        # target: extract the following components
        self.n_li = 0
        self.n_div = 0
        self.n_tr = 0
        self.n_a = 0
        self.n_img = 0
        self.n_btn = 0
        self.n_header = 0
        self.dep_header = 0
        self.b_map = 0
        self.b_pagination = 0 # or 1
        self.n_sidebar = 0
        self.n_navi = 0
        self.n_table = 0
        self.n_addr = 0
        self.n_author = 0
        self.n_time = 0

        # focus on the content
        self.keys = []

        # layout
        self.layout = None

        # load the cotent from the cachePath
        with open(cachePath, 'rb') as f:
            self.cacheNode = pickle.load(f)

        self.status_code = self.cacheNode.status_code
        self.label = self.cacheNode.label
        if self.status_code is 0:
            self.tree = html.fromstring(self.cacheNode.pageContent)

    def extract(self, stop_words):
        if self.status_code != 0:
            return

        # using xpath
        self.n_li = len(self.tree.xpath('//li'))
        self.n_div = len(self.tree.xpath('//div'))
        self.n_tr = len(self.tree.xpath('//tr'))
        self.n_table = len(self.tree.xpath('//table'))
        self.n_a = len(self.tree.xpath('//a'))
        self.n_img = len(self.tree.xpath('//img'))
        self.n_btn = len(self.tree.xpath('//button'))
        self.n_map = len(self.tree.xpath('//*[contains(@class, "map")]'))
        self.n_h1 = len(self.tree.xpath('//h1'))
        self.n_h2 = len(self.tree.xpath('//h2'))
        self.n_h3 = len(self.tree.xpath('//h3'))
        self.n_h4 = len(self.tree.xpath('//h4'))
        self.n_h5 = len(self.tree.xpath('//h5'))
        self.n_h6 = len(self.tree.xpath('//h6'))
        self.n_header = self.n_h1 + self.n_h2 + self.n_h3 + self.n_h4 + self.n_h5 + self.n_h6
        self.dep_header = int(self.n_h1 != 0) + int(self.n_h2 != 0) + int(self.n_h3 != 0) + \
                          int(self.n_h4 != 0) + int(self.n_h5 != 0) + int(self.n_h6 != 0)

        # any nodes, whose class property contains pagination
        self.b_pagination = len(self.tree.xpath('//*[contains(@class, "pagination")]'))
        self.n_navi = len(self.tree.xpath('//nav'))
        self.n_sidebar = len(self.tree.xpath('//*[contains(@id, "sidebar")]'))

        # extra, +3 field
        self.n_author = len(self.tree.xpath('//meta[@name="author"]/@content')) + \
                        len(self.tree.xpath('*[contains(@class, "author")]'))
        self.n_addr = len(self.tree.xpath('//address'))
        self.n_time = len(self.tree.xpath('//time'))

        # extract the contents, list of string sentences
        self.keys.extend(self.tree.xpath('//meta[@name="description"]/@content'))
        self.keys.extend(self.tree.xpath('//meta[@name="keywords"]/@content'))


        self.keys.extend(self.tree.xpath('//h1/text()'))
        self.keys.extend(self.tree.xpath('//h2/text()'))
        self.keys.extend(self.tree.xpath('//h3/text()'))
        self.keys.extend(self.tree.xpath('//h4/text()'))
        self.keys.extend(self.tree.xpath('//h5/text()'))
        self.keys.extend(self.tree.xpath('//h6/text()'))

        self.keys.extend(self.tree.xpath('//p/text()'))
        self.keys.extend(self.tree.xpath('//th/text()'))
        self.keys.extend(self.tree.xpath('//td/text()'))
        self.keys.extend(self.tree.xpath('//li/text()'))
        self.keys.extend(self.tree.xpath('//dt/text()'))
        self.keys.extend(self.tree.xpath('//dd/text()'))
        self.keys.extend(self.tree.xpath('//a/text()'))

        # clean the keys, tokenize, normalize
        tokens = []
        for key in self.keys:
            key = key.lower()
            tokens.extend([token.lemma for token in TextBlob(key).words])

        # load stop words, and remove stop words
        words = []
        for token in tokens:
            if not token in stop_words:
                words.append(token)

        # further clean the content,

        # get the most frequent ones
        self.messages = []
        word_counter = Counter(words)
        for tup in word_counter.most_common(30):
            self.messages.extend([tup[0]] * tup[1])

        # print self.n_sidebar, self.n_header, self.n_li, self.n_div, self.n_btn, self.n_a, self.n_img, self.n_navi, self.n_navi

class FeatureExtractor:
    # need to know 1. file list 2. where to load the tree
    def __init__(self):
        self.cacheDir = 'data'
        self.feat_path = 'feat.pkl'
        self.label_path = 'label.pkl'
        self.stop_words = set(open('stop_words.txt').read().splitlines())
        pass

    def driver(self):
        # load the cacheDir, and extract the components for TreeNode
        # !! must remove the dirty nodes, check status_code of TreeNode
        all_words = []
        self.tree_nodes = []
        self.message_list = []
        cacheFiles = os.listdir(self.cacheDir)
        cacheFiles.sort()
        for filename in cacheFiles:
            if not filename.endswith('.pkl'):
                continue

            cachePath = os.path.join(self.cacheDir, filename)
            treeNode = TreeNode(cachePath)
            if treeNode.status_code != 0:
                continue

            print filename
            treeNode.extract(self.stop_words)
            self.tree_nodes.append(treeNode)
            self.message_list.append(' '.join(treeNode.messages))

            # collect all key words
            all_words.extend(treeNode.messages)

        # build and training into tf-idf
        self.bow_transformer = CountVectorizer().fit(all_words)
        # print self.message_list
        messages_bow = self.bow_transformer.transform(self.message_list)
        # print messages_bow
        self.tfidf_transformer = TfidfTransformer().fit(messages_bow)
        self.messages_tfidf = self.tfidf_transformer.transform(messages_bow)

        self.dim = 22 + len(self.bow_transformer.vocabulary_)
        self.rows = len(self.tree_nodes)

    def extract(self):
        # allocate numpy array
        self.feats = np.zeros((self.rows, self.dim))
        self.feats[:, 22:] = self.messages_tfidf.toarray()
        self.labels = np.zeros(self.rows)

        for i in xrange(self.rows):
            treeNode = self.tree_nodes[i]
            self.feats[i][0] = treeNode.n_li
            self.feats[i][1] = treeNode.n_div
            self.feats[i][2] = treeNode.n_tr
            self.feats[i][3] = treeNode.n_a
            self.feats[i][4] = treeNode.n_img
            self.feats[i][5] = treeNode.n_btn
            self.feats[i][6] = treeNode.n_map
            self.feats[i][7] = treeNode.n_h1
            self.feats[i][8] = treeNode.n_h2
            self.feats[i][9] = treeNode.n_h3
            self.feats[i][10] = treeNode.n_h4
            self.feats[i][11] = treeNode.n_h5
            self.feats[i][12] = treeNode.n_h6
            self.feats[i][13] = treeNode.n_header
            self.feats[i][14] = treeNode.dep_header
            self.feats[i][15] = treeNode.b_pagination
            self.feats[i][16] = treeNode.n_navi
            self.feats[i][17] = treeNode.n_sidebar
            self.feats[i][18] = treeNode.n_addr
            self.feats[i][19] = treeNode.n_author
            self.feats[i][20] = treeNode.n_time
            self.feats[i][21] = treeNode.n_table

            self.labels[i] = treeNode.label

        # normalize the features by column
        self.feats = normalize(self.feats, axis = 0)


    def getFeats(self):
        return self.feats

    def getLabels(self):
        return self.labels

    def save(self):
        # save the data and save the corresponding labels
        with open(self.feat_path, 'wb') as f:
            pickle.dump(self.getFeats(), f, pickle.HIGHEST_PROTOCOL)

        with open(self.label_path, 'wb') as f:
            pickle.dump(self.getLabels(), f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    extractor = FeatureExtractor()
    extractor.driver()
    extractor.extract()
    # save the model
    extractor.save()
