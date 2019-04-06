import random
import os
import uuid # 随机取名的库

seed = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTXYZ1234567890'

# 第一种自定义名字办法
ran_name1 = []
for i in range(7):
    choice = random.choice(seed) #从种子中随机选择一个字符
    ran_name1.append(choice) # 一个随机字符加入列表
#print (''.join(ran_name1)) # "123".join("567") == "123567" 将序列中的制定字符链接生成一个新的字符串

# 第二种自定义名字办法
ran_name2 = ''.join([name2 for name2 in random.sample(seed, 6)]) # 在列表里循环 列表生成式, 从种子里取, 取6个

print(ran_name2)

# 第三种自定义名字办法
ran_name3 = random.randint(100000, 999999)
# print(ran_name3)

# 获取路径中的文件名字
path = '/mike/teacher/logo.jpg'
print(os.path.basename(path)) # 取出斜杠最后的名字==logo.jpg

# 文件名分割办法

file = 'dog.jpg'
print(file.split('.')[0]) # 放在一个列表里面 [[dog], [jpg]]
