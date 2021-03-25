import xml.etree.ElementTree as etree
import multiprocessing as mp
import bz2
import codecs
import csv
import time
import os,glob
import traceback

#WIKIPEDIA_ROOT = "/media/jeff/Data/data/wikipedia"
WIKIPEDIA_ROOT = "/home/jeff/data/wikipedia"
WIKIPEDIA_DL = os.path.join(WIKIPEDIA_ROOT, 'dl')
WORKER_REPORT = 1000
ENTIRE_TASK_REPORT = 100000

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def strip_tag_name(t):
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t

class ExtractWikipedia:
    def __init__(self, worker):
        self.totalCount = 0
        self.articleCount = 0
        self.redirectCount = 0
        self.templateCount = 0
        self.worker = worker

    def extract_file(self, path):
        start_time = time.time()

        title = None
        redirect = ""
        count = 0
        with bz2.BZ2File(path, "r") as fp:
            is_first = True
            for event, elem in etree.iterparse(fp, events=('start', 'end')):
                tname = strip_tag_name(elem.tag)
                if is_first:
                    root = elem
                    is_first = False

                if event == 'start':
                    if tname == 'page':
                        title = ''
                        id = -1
                        redirect = ''
                        inrevision = False
                        ns = 0
                    elif tname == 'revision':
                        # Do not pick up on revision id's
                        inrevision = True   
                else:
                    if tname == 'title':
                        title = elem.text
                    elif tname == 'id' and not inrevision:
                        id = int(elem.text)
                    elif tname == 'redirect':
                        redirect = elem.attrib['title']
                    elif tname == 'ns':
                        ns = int(elem.text)
                    elif tname == 'page':
                        self.totalCount += 1

                        if ns == 10:
                            self.templateCount += 1
                            self.worker.process_template(id, title)
                        elif ns == 0:
                            if len(redirect) > 0:
                                self.articleCount += 1
                                self.worker.process_redirect(id, title, redirect)
                            else:
                                self.redirectCount += 1
                                #print(f"Article: {title}")
                                self.worker.process_article(id, title)

                        title = ""
                        redirect = ""
                        ns = -100
                        if self.totalCount > 1 and (self.totalCount % WORKER_REPORT) == 0:
                            self.worker.report_progress(self.totalCount)
                            self.totalCount = 0

                        root.clear()
        self.worker.report_progress(self.totalCount)
        
class ProcessPages():
    def __init__(self, outputQueue):
        self.templates = []
        self.articles = []
        self.redirects = []
        self.outputQueue = outputQueue
        
    def process_template(self, id, title):
        self.outputQueue.put(
            {'template': [id, title] }
        )
    
    def process_article(self, id, title):
        self.outputQueue.put(
            {'article': [id, title] }
        )
    
    def process_redirect(self, id, title, redirect):
        self.outputQueue.put(
            {'redirect': [id, title, redirect] }
        )
        
    def report_progress(self, completed):
        self.outputQueue.put({"completed": completed})

def worker(inputQueue, outputQueue, config):
    done = False
    p = None
    while not done:
        try:
            path = inputQueue.get()
            #print(f"New Path: {path}")
        
            if path == "**exit**":
                p = None
                return
            p = ProcessPages(outputQueue)
            e = ExtractWikipedia(p)
            e.extract_file(path)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            if p:
                outputQueue.put({"file_complete":True})
    
ENCODING = "utf-8"

start_time = time.time()

files = glob.glob(os.path.join(WIKIPEDIA_DL, "*.bz2"))
print(f"Processing {len(files)} files")
cpus = mp.cpu_count()
print(f"Detected {cpus} cores.")
workers = cpus * 1
print(f"Using {workers} threads")

inputQueue = mp.Queue()
outputQueue = mp.Queue(50)
config = {}

processes = []
for i in range(workers):
    p = mp.Process(target=worker, args=(inputQueue, outputQueue, config))
    p.start()
    processes.append(p)
   
for file in files:
    inputQueue.put(file)
    
pathArticles = os.path.join(WIKIPEDIA_ROOT, "articles.csv")
pathRedirect = os.path.join(WIKIPEDIA_ROOT, "redirect.csv")
pathTemplate = os.path.join(WIKIPEDIA_ROOT, "template.csv")
    
total_count = 0
file_count = 0
done = False

with codecs.open(pathArticles, "w", ENCODING) as articlesFH, \
        codecs.open(pathRedirect, "w", ENCODING) as redirectFH, \
        codecs.open(pathTemplate, "w", ENCODING) as templateFH:
    articlesWriter = csv.writer(articlesFH, quoting=csv.QUOTE_MINIMAL)
    redirectWriter = csv.writer(redirectFH, quoting=csv.QUOTE_MINIMAL)
    templateWriter = csv.writer(templateFH, quoting=csv.QUOTE_MINIMAL)

    articlesWriter.writerow(['id', 'title'])
    redirectWriter.writerow(['id', 'title', 'redirect'])
    templateWriter.writerow(['id', 'title'])
    
    last_update = 0
    while file_count < len(files):
        z = outputQueue.get()
        if "completed" in z:
            total_count += z["completed"]
            current_update = int(total_count / ENTIRE_TASK_REPORT)
            if current_update != last_update:
                print(f"{current_update*ENTIRE_TASK_REPORT:,}; files: {file_count}/{len(files)}")
                last_update = current_update
        elif "file_complete" in z:
            file_count += 1
        elif "article" in z:
            articlesWriter.writerow(z['article'])
        elif "template" in z:
            templateWriter.writerow(z['template'])
        elif "redirect" in z:
            redirectWriter.writerow(z['redirect'])
       
for i in range(cpus):
    inputQueue.put("**exit**") 
      
print(f"{total_count:,}; files: {file_count}/{len(files)}")

elapsed_time = time.time() - start_time

print("waiting for workers to write remaining results")
for p in processes:
    p.join()
    
print("Elapsed time: {}".format(hms_string(elapsed_time)))   
print("done")  

#
#e = ExtractWikipedia(WIKIPEDIA_DL, p)
#e.extract_file(TEST_PATH)
#e.extract()
#p.close()
# 54249
