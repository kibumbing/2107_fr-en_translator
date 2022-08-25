import sys
sys.path.append('..')
import os
import numpy
import pickle

id_to_char = {}
char_to_id = {}
id_to_char[0] = ''
char_to_id[''] = 0
def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(file_name='europarl-v7.en.txt', seed=1984, answer=False):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    questions = []

    for line in open(file_path, 'r', encoding='UTF8'):
        idx = line.find('\n')
        questions.append(line[:idx])

    questions = questions[:50000]

    for line in range(len(questions)):
        questions[line] = questions[line].lower()
        questions[line] = questions[line].replace('(', '')
        questions[line] = questions[line].replace(')', '')
        questions[line] = questions[line].replace('?', ' ?')
        questions[line] = questions[line].replace('!', ' !')
        questions[line] = questions[line].replace('.', ' .')
        questions[line] = questions[line].replace(',', ' ,')
        questions[line] = questions[line].replace('\'', ' \'')
        if answer:
            questions[line] = '<eos> ' + questions[line]
        questions[line] = questions[line].split(' ')

    # 어휘 사전 생성
    maxCount = 0
    for i in range(len(questions)):
        q = questions[i]
        _update_vocab(q)
        Count = len(questions[i])
        if Count > maxCount:
            maxCount = Count

    # 넘파이 배열 생성
    x = numpy.zeros((len(questions), maxCount), dtype=numpy.int32)

    for index in range(len(questions)):
        for no, word in enumerate(questions[index]):
            x[index][no] = char_to_id[word]

    # 뒤섞기
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    x = x[indices]

    # 검증 데이터셋으로 10% 할당
    split_at = len(x) - len(x) // 10
    train, test = x[:split_at], x[split_at:]

    return train, test


def get_vocab():
    return char_to_id, id_to_char

x_train, x_test = load_data('europarl-v7.fr.txt')
t_train, t_test = load_data('europarl-v7.en.txt', answer=True)
char_to_id, id_to_char = get_vocab()

params = {}
params['x_train'] = x_train
params['x_test'] = x_test
params['t_train'] = t_train
params['t_test'] = t_test
params['char_to_id'] = char_to_id
params['id_to_char'] = id_to_char
pkl_file = 'en_fr_paramsTiny_50000.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)

