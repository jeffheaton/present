# Process_Wikipedia
# Copyright 2021 by Jeff Heaton, released under the MIT License
# https://github.com/jeffheaton
from urllib.parse import urlparse
import xml.etree.ElementTree as etree
import multiprocessing as mp
import bz2
import time
import os,glob
import traceback
import queue
from util import *

try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

WIKIPEDIA_URL = 'https://dumps.wikimedia.org/'
WIKIPEDIA_LANG = 'enwiki'
WORKER_REPORT = 1000
ENTIRE_TASK_REPORT = 100000
QUEUE_SIZE = 50
ENCODING = "utf-8"
GET_TIMEOUT = 1 * 60
GET_RETRY = 5

class ExtractWikipediaFile:
    """This class is spun up once per worker process."""
    def __init__(self, worker):
        self.totalCount = 0
        self.articleCount = 0
        self.redirectCount = 0
        self.templateCount = 0
        self.worker = worker

    def extract_file(self, path):

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
                    elif tname == 'text':
                        text = elem.text
                    elif tname == 'id' and not inrevision:
                        id = int(elem.text)
                    elif tname == 'redirect':
                        redirect = elem.attrib['title']
                    elif tname == 'ns':
                        ns = int(elem.text)
                    elif tname == 'page':
                        self.totalCount += 1

                        try:
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
                                    self.worker.process_article(id, title, text, path)
                        except Exception as e:
                            print(f"Error processing: {id}:{title}")
                            print(e)

                        title = ""
                        redirect = ""
                        text = ""
                        ns = -100
                        if self.totalCount > 1 and (self.totalCount % WORKER_REPORT) == 0:
                            self.worker.report_progress(self.totalCount)
                            self.totalCount = 0

                        root.clear()
        self.worker.report_progress(self.totalCount)
                
class ExtractWikipedia:
    """ This is the main controller class, it runs in the main process and aggregates results from the individual 
        processes used to distribute the workload."""
    def __init__(self, payload, path, wiki_lang=WIKIPEDIA_LANG, wiki_date=None, files=None):
        if wiki_date is None:
            wiki_date = ExtractWikipedia.find_latest_dump(path)
            if wiki_date is None:
                raise FileNotFoundError("No wiki dumps have been downloaded.")

        self.wiki_path = target_path = os.path.join(path, "dl", str(wiki_date))
        self.total_count = 0
        self.file_count = 0
        self.last_update = 0
        self.payload = payload
        self.files = files
        self.workers_running = 0
        self.workers = 0

    @staticmethod
    def find_latest_dump(path):
        target_path = os.path.join(path, "dl")
        files = glob.glob(os.path.join(target_path, "*", ""))
        files = [os.path.basename(os.path.normpath(x)) for x in files]
        files2 = []
        for f in files:
            try:
                i = int(f)
            finally:
                files2.append(f)
        files2.sort(reverse=True)
        
        if len(files2)<1:
            return None
        else:
            return files2[0]
        
    def process_single_file(self, filename=None):
        if filename==None:
            files = glob.glob(os.path.join(self.wiki_path, "*.bz2"))
            files.sort()
            filename = files[0]
        self.files = [filename]
        self.process()

    def get_event(self, event_queue):
        get_done = False
        get_retry = 0
        while not get_done:
            try:                    
                return event_queue.get(timeout=GET_TIMEOUT)
            except queue.Empty:
                get_retry += 1
                if get_retry<= GET_RETRY:
                    print(f"Queue get timeout, retry {get_retry}/{GET_RETRY}")
                else:
                    print(f"Queue timeout failed, retry {GET_RETRY} failed, exiting.")
                    get_done = True
                    return None

    def handle_event(self, evt):
        if "completed" in evt:
            self.total_count += evt["completed"]
            self.current_update = int(self.total_count / ENTIRE_TASK_REPORT)
            if self.current_update != self.last_update:
                print(f"{self.current_update*ENTIRE_TASK_REPORT:,}; files: {self.file_count}/{len(self.files)}, workers:{self.workers_running}/{self.workers}")
                self.last_update = self.current_update
        elif "file_complete" in evt:
            self.file_count += 1
        elif "**worker done**" in evt:
            self.workers_running -= 1
            print(f"Worker done: {evt['**worker done**']}")        

    def process(self):
        if self.files is None:
            self.files = glob.glob(os.path.join(self.wiki_path, "*.bz2"))
            if len(self.files)==0:
                raise FileNotFoundError(f"No wiki files located at: {self.wiki_path}")
        
        start_time = time.time()

        print(f"Processing {len(self.files)} files")
        cpus = mp.cpu_count()
        print(f"Detected {cpus} cores.")
        self.workers = cpus * 1
        print(f"Using {self.workers} threads")

        inputQueue = mp.Queue()
        outputQueue = mp.Queue(QUEUE_SIZE)


        processes = []
        for i in range(self.workers):
            config = {
                'payload': self.payload,
                'num': i
            }

            p = mp.Process(target=ExtractWikipedia.worker, args=(inputQueue, outputQueue, config))
            p.start()
            p.name = f"process-{i}"
            processes.append(p)
        self.workers_running = self.workers

        for file in self.files:
            inputQueue.put(file)            

        for i in range(self.workers*2):
            inputQueue.put("**exit**") 

        self.payload.open()
        error_exit = False
        while (self.file_count < len(self.files)) and not error_exit:
            evt = self.get_event(outputQueue)
            self.handle_event(evt)
            self.payload.handle_event(evt)

        print(f"{self.total_count:,}; files: {self.file_count}/{len(self.files)}")
        
        self.shutdown(processes, outputQueue)
        self.payload.close()
        inputQueue.close()
        outputQueue.close()

        elapsed_time = time.time() - start_time
        print("Elapsed time: {}".format(hms_string(elapsed_time)))   
        print("done")  

    def shutdown(self, processes, event_queue):
        done = False
        print("waiting for workers to write remaining results")
        while not done:
            done = True
            for p in processes:
                p.join(10)
                if p.exitcode==None: 
                    done = False
                try:                    
                    evt=event_queue.get(timeout=10)
                    self.handle_event(evt)
                except queue.Empty:
                    pass

    @staticmethod
    def get_event(workload_queue):
        get_done = False
        get_retry = 0
        while not get_done:
            try:                    
                return workload_queue.get(timeout=GET_TIMEOUT)
            except queue.Empty:
                get_retry += 1
                if get_retry<= GET_RETRY:
                    print(f"Workload get timeout, retry {get_retry}/{GET_RETRY}")
                else:
                    print(f"Workload timeout failed, retry {GET_RETRY} failed, exiting.")
                    get_done = True
                    return None

    @staticmethod
    def worker(inputQueue, outputQueue, config):
        try:
            payload_worker = config['payload'].get_worker_class(outputQueue, config)
            done = False

            while not done:
                path = inputQueue.get()

                if path != "**exit**":
                    try:
                        e = ExtractWikipediaFile(payload_worker)
                        e.extract_file(path)
                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()
                    finally:
                        outputQueue.put({"file_complete":True})
                else:
                    done = True

        finally:
            outputQueue.put({"**worker done**":config['num']})

        
        
