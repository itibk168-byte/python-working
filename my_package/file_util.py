"""
文件处理相关的模块
"""

from fileinput import close

def print_file_info(file_name):
    """
    将给定路径的文件内容输出到控制台中
    :param file_name:
    :return:
    """
    f = None
    try:
        f = open(file_name, 'r', encoding='utf-8')
    except Exception as e:
        print(f"文件异常原因是:{e}")
    else:
        content = f.read()
        print(f"文件的全部内容如下:{content}")
    finally:
        if f:           # 如果变量是None，表示False，如果有任何内容，就是True
            f.close()

def append_to_file(file_name, date):
    """
    将指定的数据追加到指定的文件中
    :param file_name:
    :param date:
    :return:
    """
    f = open(file_name, 'a', encoding='utf-8')
    f.write(date)
    f.write("\n")
    f.close()

if __name__ == "__main__":
    print_file_info("D://a")