import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# TF-IDF 모델 생성과 그래프 생성
class GraphMatrix(object):
    def __init__(self):
        self.cnt_vec = CountVectorizer()
    
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]: word for word in vocab}

# TextRank 알고리즘
class Rank(object):
    def get_ranks(self, graph, d=0.85):
        A = graph
        matrix_size = A.shape[0]
        for idx in range(matrix_size):
            A[idx, idx] = 0
            link_sum = np.sum(A[:, idx])
            if link_sum != 0:
                A[:, idx] /= link_sum
            A[:, idx] *= -d
            A[idx, idx] = 1
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)
        
        return {idx: r[0] for idx, r in enumerate(ranks)}

# TextRank Class
class TextRankModel(object):
    def __init__(self, text):
        self.graph_matrix = GraphMatrix()
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(text)
        
        self.rank = Rank()
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
    
    def get_hashtags(self, word_num=10):
        hashtags = []
        index = []
        
        for idx in self.sorted_word_rank_idx[:word_num]:
            index.append(idx)
        
        for idx in index:
            hashtags.append(self.idx2word[idx])
        
        #hashtags[0] = hashtags[0]+'브이로그'
        
        return hashtags