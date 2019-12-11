# _*_ codig utf8 _*_
import jieba

s='单词长度必须大于1'
a = jieba.cut(s)
print(list(a))