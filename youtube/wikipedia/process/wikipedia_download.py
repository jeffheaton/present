from urllib.parse import urlparse
import xml.etree.ElementTree as etree
import multiprocessing as mp
import posixpath
import time
import os
import urllib
import urllib.request
import json
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

class WikiDump:
    def __init__(self, wiki_lang=WIKIPEDIA_LANG, wiki_base_url=WIKIPEDIA_URL):
        self.wiki_lang = wiki_lang
        self.wiki_base_url = wiki_base_url
        self.wiki_url = posixpath.join(self.wiki_base_url, self.wiki_lang)

    def get_wikidump_available(self):
        index = urllib.request.urlopen(self.wiki_url).read()
        soup_index = BeautifulSoup(index, 'html.parser')
        # Find the links on the page
        return [a['href'] for a in soup_index.find_all('a') if 
                a.has_attr('href')]

    def get_latest_dump(self, dumps):
        lst = []
        for dump in dumps:
            try:
                idx = dump.index('/')
                
                if idx != -1:
                    dump = dump[:idx]
                
                lst.append(int(dump))
            except ValueError:
                pass
        lst.sort(reverse=True)
        status = self.get_status(lst[0])
        if status['jobs']['metacurrentdump']['status'] != 'done':
            return lst[1]
        return lst[0]

    def get_status(self, wiki_date):
        dump_url = posixpath.join(self.wiki_url, str(wiki_date))
        status_file = posixpath.join(dump_url, 'dumpstatus.json')
        dump_json = urllib.request.urlopen(status_file).read()
        return json.loads(dump_json)

class DownloadWikifile:
    def __init__(self, wiki_lang=WIKIPEDIA_LANG, wiki_base_url=WIKIPEDIA_URL, wiki_date=None):
        self.wiki_lang = wiki_lang
        self.wiki_base_url = wiki_base_url

        if not wiki_date:
            dump_site = WikiDump(self.wiki_lang, wiki_base_url)
            dumps = dump_site.get_wikidump_available()
            self.wiki_date = dump_site.get_latest_dump(dumps)
        else:
            self.wiki_date = wiki_date
        self.dump_url = posixpath.join(self.wiki_base_url, self.wiki_lang, str(self.wiki_date)) 

    def download(self, path):
        target_path = os.path.join(path, "dl", str(self.wiki_date))
        os.makedirs(target_path, exist_ok=True)

        
        status_file = posixpath.join(self.dump_url, 'dumpstatus.json')
        dump_json = urllib.request.urlopen(status_file).read()
        dump_status = json.loads(dump_json)
        dump_current = dump_status['jobs']['metacurrentdump']
        job_status = dump_current['status']
        
        if job_status != 'done':
            raise ValueError(f"Current Wikipedia dump is not complete yet, current dump status is: {job_status}")
            
        files = dump_current['files']
        file_count = len(files)
        for i, file in enumerate(files.keys()):
            meta = files[file]
            source_url = urllib.parse.urljoin(self.dump_url, meta['url'])
            target_file = os.path.join(target_path,file)
            if os.path.exists(target_file):
                sha1_local = sha1_file(target_file)
                if 'sha1' not in meta or sha1_local != meta['sha1']:
                    print(f"Corrupt: {file}")
                    os.remove(target_file)
                    should_download = True
                else:
                    print(f"Exists ({i}/{file_count}): {file}")
                    should_download = False
            else:
                print(f"Missing ({i}/{file_count}): {file}")
                should_download = True
                
            if should_download:
                try:
                    # https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
                    # https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
                    print(f"Begin download: {file}")
                    start_time = time.time()
                    urllib.request.urlretrieve(source_url, target_file)
                    elapsed_time = time.time() - start_time
                    print(f"Downloaded: {file}, time: {hms_string(elapsed_time)}")
                except urllib.error.URLError as e:
                    try:
                        os.remove(target_file)
                    finally:
                        print(f"Download URL Error: {file}")
                except ConnectionResetError:
                    try:
                        os.remove(target_file)
                    finally:
                        print(f"Download Error: {file}")

        
