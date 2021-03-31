from multiprocessing import freeze_support
from process_wikipedia import *

#WIKIPEDIA_ROOT = "/home/jeff/data/wikipedia"
WIKIPEDIA_ROOT = "C:\\Users\\jeffh\\data\\wikipedia\\"
WIKIPEDIA_DL = os.path.join(WIKIPEDIA_ROOT, 'dl')

class ProcessPagesWorker():
    def __init__(self, config, outputQueue):
        self.config = config
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

class ProcessPages:
    def __init__(self, output_path): 
        self.output_path = output_path       
        
    def open(self):
        pathArticles = os.path.join(self.output_path, "article.csv")
        pathRedirect = os.path.join(self.output_path, "redirect.csv")
        pathTemplate = os.path.join(self.output_path, "template.csv")
        
        self.articles_fp = codecs.open(pathArticles, "w", ENCODING)
        self.redirect_fp = codecs.open(pathRedirect, "w", ENCODING)
        self.template_fp = codecs.open(pathTemplate, "w", ENCODING)
    
        self.articlesWriter = csv.writer(self.articles_fp, quoting=csv.QUOTE_MINIMAL)
        self.redirectWriter = csv.writer(self.redirect_fp, quoting=csv.QUOTE_MINIMAL)
        self.templateWriter = csv.writer(self.template_fp, quoting=csv.QUOTE_MINIMAL)
        
        self.articlesWriter.writerow(['id', 'title'])
        self.redirectWriter.writerow(['id', 'title', 'redirect'])
        self.templateWriter.writerow(['id', 'title'])

    def close(self):
        self.articles_fp.close()
        self.redirect_fp.close()
        self.template_fp.close()

    def handle_event(self, evt):

        if "article" in evt:
            self.articlesWriter.writerow(evt['article'])
        elif "template" in evt:
            self.templateWriter.writerow(evt['template'])
        elif "redirect" in evt:
            self.redirectWriter.writerow(evt['redirect'])
            
    def get_worker_class(self, outputQueue, config):
        return ProcessPagesWorker(config, outputQueue)
    
if __name__ == '__main__':
    freeze_support()
    wiki = ExtractWikipedia(
        ProcessPages(WIKIPEDIA_ROOT), # where you want the extracted Wikipedia files to go
        WIKIPEDIA_DL #Location you downloaded Wikipedia to
    )
    wiki.process()
