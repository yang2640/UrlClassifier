from lxml import html
import requests
from lxml.html.clean import Cleaner
import os
import cPickle as pickle
import pandas as pd

class CacheNode:
    def __init__(self, id, url, category, pageContent, status_code):
        self.id = id
        self.url = url
        self.category = category
        self.pageContent = pageContent
        self.status_code = status_code
        self.label = self.category2label(category)

    def category2label(self, category):
        label_map = {'DOCUMENTATION': 0,
                     'INFOGRAPHIC': 1,
                     'PRESS RELEASE': 2,
                     'PRODUCT CATEGORY': 3,
                     'PRODUCT COMPARISON': 4,
                     'PRODUCT DETAIL': 5,
                     'STORE LOCATOR': 6,
                     'WHITEPAPER': 7}

        return label_map[category]

class Crawler:
    def __init__(self, id, url, category):
        self.id = id
        self.url = url
        self.category = category
        self.pageContent = None
        self.dirPath = 'data'
        self.status_code = 0

        user_agent = {
            'User-agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.52 Safari/537.17'}

        try:
            response = requests.get(url, headers=user_agent, timeout=10)

            # Only store the content if the page load was successful
            if response.ok:
                # clean html tags
                cleaner = Cleaner(style=True, links=True)
                self.pageContent = cleaner.clean_html(response.content)
                # self.tree = html.fromstring(self.pageContent)
                # self.texts = tree.xpath('//text()')
                print 'processing id %05d' % (self.id)
            else:
                self.status_code = response.status_code
                print 'Error processing URL: %s\nstatus code: %d' % (url, response.status_code)

        except Exception as inst:
            self.status_code = -1
            print inst
            print 'Error processing URL: %s\n' % (url)


    def getUrl(self):
        return self.url

    def save(self):
        # build the cache node
        cacheNode = CacheNode(self.id, self.url, self.category, self.pageContent, self.status_code)
        # save each url as page content representation
        path = os.path.join(self.dirPath, '%05d.pkl' % (self.id))
        with open(path, 'wb') as f:
           pickle.dump(cacheNode, f, pickle.HIGHEST_PROTOCOL)

# read the file list and crawl all the webpages
if __name__ == '__main__':
    # read in the file list
    dataset = 'handout_urls_with_categories.tsv'
    dataframe = pd.read_csv(dataset, delimiter='\t')
    # enumerate all the webpages and crawl the data
    for id, url in enumerate(dataframe.page_url):
        # capture url prefix missing
        if not url.startswith('http://'):
            url = 'http://%s' % (url)
        crawler = Crawler(id, url, dataframe.page_category[id])
        crawler.save()


