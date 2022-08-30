from multiprocessing import freeze_support
from wikipedia_process import *
import wikitextparser as wtp
import ntpath

from wikipedia_process import *

#WIKIPEDIA_ROOT = "/Users/jheaton/jth/wikipedia"
#WIKIPEDIA_ROOT = "/home/jeff/data/wikipedia"
#WIKIPEDIA_ROOT = "C:\\jth\\data\\wikipedia\\"
WIKIPEDIA_ROOT = "/Users/jeff/data/wiki"

REMOVE_DUPS = True

TOKEN_INFOBOX = "Infobox "

class IndexWikipediaWorker():
    def __init__(self, config, outputQueue):
        self.config = config
        self.outputQueue = outputQueue
        
    def process_template(self, id, title):
        self.outputQueue.put(
            {'template': [id, title] }
        )
    
    def process_article(self, id, title, text, path):
        page = wtp.parse(text)
        tlist = set()
        o = []
        iname = []
        for t in page.templates:
            tname = t.name.strip()
            if tname not in tlist or not REMOVE_DUPS:
                tlist.add(tname)
                o.append([id, tname])

            if tname.startswith(TOKEN_INFOBOX):
                iname.append(tname[len(TOKEN_INFOBOX):].strip())


        if len(iname)==0:
            iname = ""
        elif len(iname)==1:
            iname = iname[0]
        else:
            iname = str(iname)
                
        self.outputQueue.put({'article_template':o})

        llist = set()
        o = []
        for l in page.wikilinks:
            ltitle = l.title.strip()
            if ltitle not in llist or not REMOVE_DUPS:
                llist.add(ltitle)
                o.append([id, ltitle])

        self.outputQueue.put({'article_link':o})

        filename = ntpath.basename(path)
        self.outputQueue.put(
            {'article': [id, title, iname, filename] }
        )

    
    def process_redirect(self, id, title, redirect):
        self.outputQueue.put(
            {'redirect': [id, title, redirect] }
        )
        
    def report_progress(self, completed):
        self.outputQueue.put({"completed": completed})

class IndexWikipedia:
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
        
        self.articlesWriter.writerow(['article_id', 'title', 'infobox', 'file'])
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
        return IndexWikipediaWorker(config, outputQueue)
    
if __name__ == '__main__':
    freeze_support()
    wiki = ExtractWikipedia(
        payload=IndexWikipedia(WIKIPEDIA_ROOT), # where you want the extracted Wikipedia files to go
        path=WIKIPEDIA_ROOT #Location you downloaded Wikipedia to
    )
    #wiki.process()
    wiki.process_single_file()

