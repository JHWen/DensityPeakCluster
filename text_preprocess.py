import jieba


def load_stop_words(stop_words_path):
    stop_words = set()
    with open(stop_words_path, mode='r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                stop_words.add(line.strip())
    return stop_words


def segment_text(file_to_segment, file_to_save):
    stop_words = load_stop_words('./data/spam_mail_data/stopwords.txt')
    file_write = open(file_to_save, mode='w', encoding='utf-8')
    with open(file_to_segment, mode='r', encoding='utf-8') as f:
        for line in f:
            cut_words = [word for word in jieba.cut(line.strip()) if word not in stop_words]
            file_write.write(' '.join(cut_words) + '\n')
    file_write.close()


if __name__ == '__main__':
    # 对正常邮件分词
    segment_text('./data/spam_mail_data/ham_5000.utf8', './data/spam_mail_data/ham_5000_segment.utf8')
    # 对垃圾邮件分词
    segment_text('./data/spam_mail_data/spam_5000.utf8', './data/spam_mail_data/spam_5000_segment.utf8')

