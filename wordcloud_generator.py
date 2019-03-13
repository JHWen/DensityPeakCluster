from wordcloud import WordCloud
import matplotlib.pyplot as plt


def replace_space_in_text(text_path, text_save_path):
    file_to_save = open(text_save_path, mode='w', encoding='utf-8')
    with open(text_path, mode='r', encoding='utf-8') as lines:
        for line in lines:
            words = [word.strip() for word in line.strip().split(' ') if len(word.strip()) > 0]
            file_to_save.write(' '.join(words))
            file_to_save.write('\n')
    file_to_save.close()


if __name__ == '__main__':
    text = open('./data/spam_mail_data/ham_5000_segment_replace_space.utf8', mode='r', encoding='utf-8').read()
    # C:\Windows\Fonts\simkai.ttf
    font_path = 'C:/Windows/Fonts/simkai.ttf'
    wc = WordCloud(font_path=font_path, width=1920, height=1080, max_words=300, collocations=False,
                   background_color='white')
    word_frequencies = wc.process_text(text)
    word_frequencies_list = sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True)
    print(word_frequencies_list)
    picture_data = wc.generate(text)
    plt.axis('off')
    plt.imshow(picture_data)
    plt.show()
    # replace_space_in_text('./data/spam_mail_data/ham_5000_segment.utf8',
    #                       './data/spam_mail_data/ham_5000_segment_replace_space.utf8')
