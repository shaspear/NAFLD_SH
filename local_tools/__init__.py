# 导入模块
from . import my_dataset
from . import common_tools

# 定义变量
version = '1.0'

# 定义函数
def greet():
    print('Hello, welcome to the Python subpackage!')

# 指定可导入的模块或变量列表
__all__ = ['my_dataset', 'common_tools', 'version', 'greet']