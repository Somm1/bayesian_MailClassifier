import os
import operator
import nltk


def read_email(path):  # 读取邮件文件
    files = os.listdir(path)  # 获得文件夹内所有的文件名
    s = []
    for file in files:
        f = open(path + "/" + file, 'r', errors='ignore')  # 打开这个邮件，直接忽略未知字符
        email = f.read()
        s.append(email)
        f.close()
    return s


# 对邮件内容预处理（除去标点符号，回车换行符等等）
def email_clean(email):
    punctuations = """,:;.<>()*&^%$%#@!'";~`[]{}|、\\/~+_-=?"""
    for letter in range(len(email)):
        # 替换掉回车，换行符
        email[letter] = email[letter].replace(
            '\n',
            ' ').replace(
            '\r',
            ' ').replace(
            '\t',
            ' ')
    for letter in range(len(email)):
        # 全部改为小写字符
        email[letter] = email[letter].lower()
    for letter in range(len(email)):
        # 替换标点符号
        for punctuation in punctuations:
            email[letter] = email[letter].replace(punctuation, ' ')
    return email


# 统计词频
def count_word(email_word):
    single_email_word = []
    # 去除重复词汇
    for i in range(len(email_word)):
        single_email_word.append(list(set(email_word[i])))
    word_prob = {}  # 新建一个字典
    # 每封邮件中，出现过，就加1
    for emailList in range(len(single_email_word)):
        # 一封一封来统计
        for word in single_email_word[emailList]:
            word_prob[word] = word_prob.get(word, 0) + 1  # 如果这个词已经出现过，那就加0；新出现的词，则加1
    sorted_word_prob = sorted(
        word_prob.items(),
        key=operator.itemgetter(1),
        reverse=True)
    return sorted_word_prob


# 删除停用词
def delete_stop_word(email_word):
    for letter in range(len(email_word)):
        for word in iter(email_word[letter]):  # 生成一个迭代器
            while word in nltk.corpus.stopwords.words('english'):
                if word in email_word[letter]:
                    email_word[letter].remove(word)
                elif word not in email_word[letter]:
                    break
    return email_word


def bayes_filter(test_word, SpamWord_dic, HamWord_dic):
    test_word_prob = {}  # 新建一个字典
    single_word_email = list(set(test_word))
    # 统计测试数据中的词频
    for word in single_word_email:
        test_word_prob[word] = test_word_prob.get(word, 0) + 1
    wordlist = []
    for word in test_word_prob.keys():
        wordlist.append(word)
        SpamWord_dic[word] = SpamWord_dic.get(
            word, 10)  # 若训练集中这个单词从未出现，姑且算它100次
        HamWord_dic[word] = HamWord_dic.get(word, 10)
    # 计算各种概率
    emailNum_spam = len(os.listdir(r"train\spam"))  # 垃圾邮件的封数
    emailNum_ham = len(os.listdir(r"train\ham"))  # 正常邮件的封数
    P_spam = operator.truediv(emailNum_spam, (emailNum_spam + emailNum_ham))  # 垃圾邮件的概率
    P_ham = operator.truediv(emailNum_ham, (emailNum_spam + emailNum_ham))  # 正常邮件的概率
    # k个词的概率
    P_word = {}
    for word in wordlist:
        P_word[word] = operator.truediv((SpamWord_dic[word] + HamWord_dic[word]), (emailNum_spam + emailNum_ham))
    # 垃圾邮件中各个词的概率
    P_word_inSpam = {}  # k个词在垃圾邮件中的概率
    for word in wordlist:
        P_word_inSpam[word] = operator.truediv(SpamWord_dic[word], emailNum_spam)
    # 正常邮件中各个词的概率
    P_word_inHam = {}
    for word in wordlist:
        P_word_inHam[word] = operator.truediv(HamWord_dic[word], emailNum_ham)
    # 贝叶斯概率公式
    P_bayes_spam = P_spam
    for word in wordlist:
        P_bayes_spam = operator.mul(P_bayes_spam, P_word_inSpam[word])
        P_bayes_spam = operator.truediv(P_bayes_spam, P_word[word])
    P_bayes_ham = P_ham
    for word in wordlist:
        P_bayes_ham = operator.mul(P_bayes_ham, P_word_inHam[word])
        P_bayes_ham = operator.truediv(P_bayes_ham, P_word[word])
    rate = operator.truediv(P_bayes_ham, P_bayes_spam)
    # rate>1则为正常邮件，<1则为垃圾邮件
    return rate


# nltk.download('stopwords')  # 下载停用词
train_spam = read_email(r"train\spam")  # 训练集的垃圾邮件
train_ham = read_email(r"train\ham")  # 训练集的正常邮件
test_spam = read_email(r"test\spam")  # 测试集的垃圾邮件
test_ham = read_email(r"test\ham")  # 测试集的正常邮件
# 得到len(train_spam)维的列表，其中每一行是一封邮件的内容
# 对邮件进行预处理（大小写，标点符号）
email_clean(train_spam)
email_clean(train_ham)
email_clean(test_spam)
email_clean(test_ham)
# 对每一封邮件分词
train_spam_word = []
train_ham_word = []
test_spam_word = []
test_ham_word = []
for i in range(len(train_spam)):
    train_spam_word.append(train_spam[i].split())
for i in range(len(train_ham)):
    train_ham_word.append(train_ham[i].split())
for i in range(len(test_spam)):
    test_spam_word.append(test_spam[i].split())
for i in range(len(test_ham)):
    test_ham_word.append(test_ham[i].split())
# train_spam_word形如[ ['abc','def'],['efd'],[...]... ]
# 下面删除停用词
delete_stop_word(train_spam_word)
delete_stop_word(train_ham_word)
delete_stop_word(test_spam_word)
delete_stop_word(test_ham_word)
# 统计词出现的频数
SpamWord_prob = count_word(train_spam_word)  # 垃圾邮件中关键词频数
HamWord_prob = count_word(train_ham_word)  # 正常邮件中关键词频数
SpamWord_dic = {}  # 改回字典类型
HamWord_dic = {}
for i in range(len(SpamWord_prob)):
    SpamWord_dic[SpamWord_prob[i][0]] = SpamWord_prob[i][1]
for i in range(len(HamWord_prob)):
    HamWord_dic[HamWord_prob[i][0]] = HamWord_prob[i][1]
# 朴素贝叶斯算法
i = 0
correct1 = 0
for i in range(len(test_spam_word)):  # 垃圾邮件辨别
    spam = bayes_filter(test_spam_word[i], SpamWord_dic, HamWord_dic)
    if spam < 1:
        correct1 = operator.add(correct1, 1)
    else:
        continue
spam_rate = operator.truediv(correct1, len(test_spam_word))
correct2 = 0
for i in range(len(test_ham_word)):
    ham = bayes_filter(test_ham_word[i], SpamWord_dic, HamWord_dic)
    if ham >= 1:
        correct2 = operator.add(correct2, 1)
    else:
        continue
ham_rate = operator.truediv(correct2, len(test_ham_word))
all_rate = operator.truediv((correct1 + correct2), (len(test_ham_word)+len(test_spam_word)))
print('垃圾邮件分辨正确率：')
print(spam_rate)
print('正常邮件分辨正确率：')
print(ham_rate)
print('总正确率：')
print(all_rate)