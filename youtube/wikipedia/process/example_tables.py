from multiprocessing import freeze_support
from process_wikipedia import *
import wikitextparser as wtp

#WIKIPEDIA_ROOT = "/Users/jheaton/jth/wikipedia"
#WIKIPEDIA_ROOT = "/home/jeff/data/wikipedia"
#WIKIPEDIA_ROOT = "C:\\jth\\data\\wikipedia\\"
WIKIPEDIA_ROOT = "/Users/jeff/data/wiki"

REMOVE_DUPS = True

class TablesWorker():
    def __init__(self, config, outputQueue):
        self.config = config
        self.outputQueue = outputQueue
        
    def process_template(self, id, title):
        self.outputQueue.put(
            {'template': [id, title] }
        )
    
    def process_article(self, id, title, text):
        if "{{Infobox" in text:
            print(title)
        self.outputQueue.put(
            {'article': [id, title] }
        )

        #page = wtp.parse(text)
        #tlist = set()
        #o = []
        #for t in page.templates:
        #    tname = t.name.strip()
        #    if tname not in tlist or not REMOVE_DUPS:
        #        tlist.add(tname)
        #        o.append([id, tname])
        #        
        #self.outputQueue.put({'article_template':o})

        #llist = set()
        #o = []
        #for l in page.wikilinks:
        #    ltitle = l.title.strip()
        #    if ltitle not in llist or not REMOVE_DUPS:
        #        llist.add(ltitle)
        #        o.append([id, ltitle])

        #self.outputQueue.put({'article_link':o})

    
    def process_redirect(self, id, title, redirect):
        self.outputQueue.put(
            {'redirect': [id, title, redirect] }
        )
        
    def report_progress(self, completed):
        self.outputQueue.put({"completed": completed})

class TablesPages:
    def __init__(self, output_path): 
        self.output_path = output_path       
        
    def open(self):
        pass

    def close(self):
        pass

    def handle_event(self, evt):
        pass
        
            
    def get_worker_class(self, outputQueue, config):
        return TablesWorker(config, outputQueue)
    
if __name__ == '__main__':
    freeze_support()
    wiki = ExtractWikipedia(
        payload=TablesPages(WIKIPEDIA_ROOT), # where you want the extracted Wikipedia files to go
        path=WIKIPEDIA_ROOT #Location you downloaded Wikipedia to
    )
    wiki.process()

