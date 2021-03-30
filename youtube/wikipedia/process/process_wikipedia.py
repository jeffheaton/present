# Process_Wikipedia
# Copyright 2021 by Jeff Heaton, released under the MIT License
# https://github.com/jeffheaton
import bs4
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import xml.etree.ElementTree as etree
import multiprocessing as mp
import posixpath
import bz2
import codecs
import csv
import time
import os,glob
import traceback
import urllib
import json
import sys
import hashlib

WIKIPEDIA_URL = 'https://dumps.wikimedia.org/'
WIKIPEDIA_LANG = 'enwiki'
WORKER_REPORT = 1000
ENTIRE_TASK_REPORT = 100000
QUEUE_SIZE = 50
ENCODING = "utf-8"
BUF_SIZE = 65536

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

def get_wikidump_available(wiki_url=WIKIPEDIA_URL,wiki_lang=WIKIPEDIA_LANG):
    base_url = posixpath.join(wiki_url, wiki_lang)
    index = urllib.request.urlopen(base_url).read()
    soup_index = BeautifulSoup(index, 'html.parser')
    # Find the links on the page
    return [a['href'] for a in soup_index.find_all('a') if 
             a.has_attr('href')]

def get_latest_wikidump(dumps):
    lst = []
    for dump in dumps:
        try:
            idx = dump.index('/')
            
            if idx != -1:
                dump = dump[:idx]
            
            lst.append(int(dump))
        except ValueError:
            pass
        
    return max(lst)

def sha1_file(path):
    sha1 = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

def wikidump_download(target_path, wiki_url=WIKIPEDIA_URL,wiki_lang=WIKIPEDIA_LANG, timestamp=None):
    if not timestamp:
        timestamp = get_latest_wikidump(get_wikidump_available(wiki_url,wiki_lang))
    
    dump_url = posixpath.join(wiki_url, wiki_lang, str(timestamp))
    print(dump_url)
    
    status_file = posixpath.join(dump_url, 'dumpstatus.json')
    dump_json = urllib.request.urlopen(status_file).read()
    dump_status = json.loads(dump_json)
    dump_current = dump_status['jobs']['metacurrentdump']
    job_status = dump_current['status']
    
    if job_status != 'done':
        raise ValueError(f"Current article dump status: [job_status]")
        
    files = dump_current['files']
    for file in files.keys():
        meta = files[file]
        source_url = urllib.parse.urljoin(wiki_url, meta['url'])
        target_file = os.path.join(target_path,file)
        if os.path.exists(target_file):
            sha1_local = sha1_file(target_file)
            if sha1_local != meta['sha1']:
                print(f"Corrupt: {file}")
                os.remove("file")
                should_download = True
            else:
                print(f"Exists: {file}")
                should_download = False
        else:
            print(f"Missing: {file}")
            should_download = True
            
        if should_download:
            try:
                urllib.request.urlretrieve(source_url, target_file)
                print(f"Downloaded: {file}")
            except urllib.error.URLError as e:
                try:
                    os.remove(target_file)
                finally:
                    print(f"Download Error: {file}")


class ExtractWikipediaFile:
    """This class is spun up once per worker process."""
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
                
class ExtractWikipedia:
    """ This is the main controller class, it runs in the main process and aggregates results from the individual 
        processes used to distribute the workload."""
    def __init__(self, payload, path):
        self.wiki_path = path
        self.total_count = 0
        self.file_count = 0
        self.last_update = 0
        self.payload = payload
    
    def process(self):
        start_time = time.time()
        self.files = glob.glob(os.path.join(self.wiki_path, "*.bz2"))
        print(f"Processing {len(self.files)} files")
        cpus = mp.cpu_count()
        print(f"Detected {cpus} cores.")
        workers = cpus * 1
        print(f"Using {workers} threads")

        inputQueue = mp.Queue()
        outputQueue = mp.Queue(QUEUE_SIZE)
        config = {
            'payload': self.payload
        }

        processes = []
        for i in range(workers):
            p = mp.Process(target=ExtractWikipedia.worker, args=(inputQueue, outputQueue, config))
            p.start()
            processes.append(p)

        for file in self.files:
            inputQueue.put(file)

        while self.file_count < len(self.files):
            evt = outputQueue.get()
            
            if "completed" in evt:
                self.total_count += evt["completed"]
                self.current_update = int(self.total_count / ENTIRE_TASK_REPORT)
                if self.current_update != self.last_update:
                    print(f"{self.current_update*ENTIRE_TASK_REPORT:,}; files: {self.file_count}/{len(self.files)}")
                    self.last_update = self.current_update
            elif "file_complete" in evt:
                self.file_count += 1
            
            self.payload.handle_event(evt)

        for i in range(cpus):
            inputQueue.put("**exit**") 

        print(f"{self.total_count:,}; files: {self.file_count}/{len(self.files)}")

        elapsed_time = time.time() - start_time

        print("waiting for workers to write remaining results")
        for p in processes:
            p.join()

        print("Elapsed time: {}".format(hms_string(elapsed_time)))   
        print("done")  
        
    @staticmethod
    def worker(inputQueue, outputQueue, config):
        payload_worker = config['payload'].get_worker_class(outputQueue, config)
        done = False

        while not done:
            try:
                path = inputQueue.get()

                if path == "**exit**":
                    return

                e = ExtractWikipediaFile(payload_worker)
                e.extract_file(path)
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
            finally:
                outputQueue.put({"file_complete":True})
        
        
