from multiprocessing import freeze_support
from process_wikipedia import *
import wikitextparser as wtp

WIKIPEDIA_ROOT = "/Users/jheaton/jth/wikipedia"
#WIKIPEDIA_ROOT = "/home/jeff/data/wikipedia"
#WIKIPEDIA_ROOT = "C:\\Users\\jeffh\\data\\wikipedia\\"

class CatsAndLinksWorker():
    def __init__(self, config, outputQueue):
        self.config = config
        self.outputQueue = outputQueue
        
    def process_template(self, id, title):
        self.outputQueue.put(
            {'template': [id, title] }
        )
    
    def process_article(self, id, title, text):
        self.outputQueue.put(
            {'article': [id, title] }
        )

        page = wtp.parse(text)
        tlist = set()
        o = []
        for t in page.templates:
            tname = t.name.strip()
            if tname not in tlist:
                tlist.add(tname)
                o.append([id, tname])
                
        self.outputQueue.put({'article_template':o})

        llist = set()
        o = []
        for l in page.wikilinks:
            ltitle = l.title.strip()
            if ltitle not in llist:
                llist.add(ltitle)
                o.append([id, ltitle])

        self.outputQueue.put({'article_link':o})

    
    def process_redirect(self, id, title, redirect):
        self.outputQueue.put(
            {'redirect': [id, title, redirect] }
        )
        
    def report_progress(self, completed):
        self.outputQueue.put({"completed": completed})

class CatsAndLinksPages:
    def __init__(self, output_path): 
        self.output_path = output_path       
        
    def open(self):
        pathArticles = os.path.join(self.output_path, "article.csv")
        pathArticleTemplates = os.path.join(self.output_path, "articleTemplate.csv")
        pathArticleLinks = os.path.join(self.output_path, "articleLink.csv")
        pathRedirect = os.path.join(self.output_path, "redirect.csv")
        pathTemplate = os.path.join(self.output_path, "template.csv")
        
        self.articles_fp = codecs.open(pathArticles, "w", ENCODING)
        self.articleTemplate_fp = codecs.open(pathArticleTemplates, "w", ENCODING)
        self.articleLink_fp = codecs.open(pathArticleLinks, "w", ENCODING)
        self.redirect_fp = codecs.open(pathRedirect, "w", ENCODING)
        self.template_fp = codecs.open(pathTemplate, "w", ENCODING)
    
        self.articlesWriter = csv.writer(self.articles_fp, quoting=csv.QUOTE_MINIMAL)
        self.articleTemplatesWriter = csv.writer(self.articleTemplate_fp, quoting=csv.QUOTE_MINIMAL)
        self.articleLinkWriter = csv.writer(self.articleLink_fp, quoting=csv.QUOTE_MINIMAL)
        self.redirectWriter = csv.writer(self.redirect_fp, quoting=csv.QUOTE_MINIMAL)
        self.templateWriter = csv.writer(self.template_fp, quoting=csv.QUOTE_MINIMAL)
        
        self.articlesWriter.writerow(['article_id', 'title'])
        self.articleTemplatesWriter.writerow(['article_id', 'template_name'])
        self.articleLinkWriter.writerow(['source_article_id', 'target_article_name'])
        self.redirectWriter.writerow(['article_id', 'title', 'redirect'])
        self.templateWriter.writerow(['article_id', 'title'])

    def close(self):
        self.articles_fp.close()
        self.articleTemplate_fp.close()
        self.articleLink_fp.close()
        self.redirect_fp.close()
        self.template_fp.close()

    def handle_event(self, evt):

        if "article" in evt:
            self.articlesWriter.writerow(evt['article'])
        elif "article_template" in evt:
            self.articleTemplatesWriter.writerows(evt['article_template'])
        elif "article_link" in evt:
            self.articleLinkWriter.writerows(evt['article_link'])
        elif "template" in evt:
            self.templateWriter.writerow(evt['template'])
        elif "redirect" in evt:
            self.redirectWriter.writerow(evt['redirect'])
            
    def get_worker_class(self, outputQueue, config):
        return CatsAndLinksWorker(config, outputQueue)
    
if __name__ == '__main__':
    freeze_support()
    wiki = ExtractWikipedia(
        payload=CatsAndLinksPages(WIKIPEDIA_ROOT), # where you want the extracted Wikipedia files to go
        path=WIKIPEDIA_ROOT #Location you downloaded Wikipedia to
    )
    wiki.process()

