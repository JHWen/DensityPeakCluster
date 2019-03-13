from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time


def calc_cosine_distance(vec1, vec2):
    num = np.dot(vec1, vec2.T)
    denum = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return 1 - sim[0, 0]


if __name__ == '__main__':
    cut_words_list = list()
    count = 500
    with open('./data/spam_mail_data/spam_5000_segment.utf8', mode='r', encoding='utf-8') as f:
        i = 0
        for line in f:
            cut_words_list.append(line.strip())
            i += 1
            if i >= count:
                break
    spam_mail_num = len(cut_words_list)
    print('spam mail num:' + str(spam_mail_num))
    with open('./data/spam_mail_data/ham_5000_segment.utf8', mode='r', encoding='utf-8') as f:
        i = 0
        for line in f:
            cut_words_list.append(line.strip())
            i += 1
            if i >= count:
                break
    ham_mail_num = len(cut_words_list) - spam_mail_num
    print('spam mail num:' + str(ham_mail_num))
    vectorizer = TfidfVectorizer(min_df=10, max_df=0.6)
    vectorizer_model = vectorizer.fit(cut_words_list)
    tfidf_matrix = vectorizer_model.fit_transform(cut_words_list)
    print(len(vectorizer_model.vocabulary_))
    print(vectorizer_model.vocabulary_)
    # print(tfidf_matrix)
    tfidf_data = tfidf_matrix.todense()
    print(tfidf_data.shape)
    print(tfidf_matrix.shape)
    start = time.time()
    fo = open('./data/spam_mail_data/spam_distance_1000.dat', 'w', encoding='utf-8')
    for i in range(tfidf_data.shape[0] - 1):
        for j in range(i, tfidf_data.shape[0]):
            distance = calc_cosine_distance(tfidf_data[i], tfidf_data[j])
            fo.write(str(i + 1) + ' ' + str(j + 1) + ' ' + str(distance) + '\n')
    fo.close()
    end = time.time()
    print('计算距离共花费时间:%.2f' % (end - start))
