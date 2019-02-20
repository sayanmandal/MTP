from shutil import copy2
import os


def copy_files(A,B):
    files = sorted(os.listdir(A))
    cnt = 0
    for f in files:
        cnt += 1
        copy2(A+f,B)
        if cnt == 1000:
            break
if __name__ == '__main__':
    copy_files("Semantic/","Labels/")
