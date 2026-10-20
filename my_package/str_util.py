"""
字符串相关的工具模块
"""

def str_reverse(s):
    """
    接受传入字符串，将字符串反转返回
    :param s: 传入字符串
    :return:
    """
    return s[::-1]

# print(str_reverse("hello"))

def substr(s, x, y):
    """
    按照下标x和y，对字符串进行切片
    :param s: 传入字符串
    :param x:
    :param y:
    :return:
    """
    return s[x:y]


if __name__ == "__main__":
    print(str_reverse("abc"))
    print(substr("abcd", 2, 4))
