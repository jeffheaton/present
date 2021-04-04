from multiprocessing import freeze_support
from process_wikipedia import *

#WIKIPEDIA_ROOT = "/Users/jheaton/jth/wikipedia"
#WIKIPEDIA_ROOT = "/home/jeff/data/wikipedia"
WIKIPEDIA_ROOT = "C:\\jth\\data\\wikipedia\\"
WIKIPEDIA_DL = os.path.join(WIKIPEDIA_ROOT, 'dl')

if __name__ == '__main__':
    freeze_support()
    dl=DownloadWikifile()
    dl.download(WIKIPEDIA_ROOT)