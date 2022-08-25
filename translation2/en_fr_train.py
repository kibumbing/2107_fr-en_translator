import sys

sys.path.append('..')
import matplotlib.pyplot as plt
from common_2.optimizer import Adam
from common_2.trainer import Trainer
from common_2.util import eval_seq2seq, to_gpu
from en_fr_seq2seq import Seq2seq, PeekySeq2seq, AttentionSeq2seq
import pickle
from common_2 import config
import numpy
config.GPU = True
from common_2.np import *

# 성능이 충분한 컴퓨터는 위 파일 사용가능, 아래 파일은 코드 작동만 되며, 제대로된 결과를 내지 못함.
# pkl_file = 'en_fr_paramsTiny_50000.pkl'
pkl_file = 'en_fr_paramsTiny_50.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    x_train = to_gpu(params['x_train'])
    x_test = to_gpu(params['x_test'])
    t_train = to_gpu(params['t_train'])
    t_test = to_gpu(params['t_test'])
    char_to_id = params['char_to_id']
    id_to_char = params['id_to_char']

print("load  data")

is_reverse = True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

vocab_size = len(char_to_id)
wordvec_size = 128
hidden_size = 128
batch_size = 64
max_epoch = 50
max_grad = 5.0

# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
print("create  model")
acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train,max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbos = i < 10
        correct_num += eval_seq2seq(model, question,correct, id_to_char, verbos, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('검증 정확도 %.3f%%' % (acc * 100))