import shutil
import numpy as np


MAX_STR_LEN, _ = shutil.get_terminal_size()
TITLE_STR_BUFFER_HW = int(MAX_STR_LEN // 20)


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def make_header(title):
    MAX_STR_LEN, _ = shutil.get_terminal_size()
    row_htag = '▓' * MAX_STR_LEN
        
    r1 = row_htag
    
    nhtag = MAX_STR_LEN - len(title) - 2*TITLE_STR_BUFFER_HW
    r2_l = '▓' * int(nhtag//2)
    r2_r = '▓' * int(nhtag//2)
    if nhtag % 2 == 1:
        r2_r += '▓'
        
        
    title_txt = ''
    n = 0
    for i, t in enumerate(title):
        
        if n < 1:
            title_txt += color.BLUE + t + color.END
        else:
            title_txt += color.YELLOW + t + color.END
        
        n += 1
        
        if n == 2:
            n = 0
        
    r2 = r2_l + ' '*TITLE_STR_BUFFER_HW + color.BOLD + title_txt + color.END + ' '*TITLE_STR_BUFFER_HW + r2_r
    
    r3 = row_htag
    
    print(r1)
    print(r2)
    print(r3)
    
    return
