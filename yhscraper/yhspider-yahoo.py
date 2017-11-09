import scrapy

import sys
import datetime
import csv

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector

from yhscraper.items import YhscraperItem

reload(sys)
sys.setdefaultencoding('utf-8')
class YhSpider(CrawlSpider):
	name = "yh" # Unique, name of the spider
	allowed_domains = ['yahoo.com']
	start_urls = ['https://news.search.yahoo.com/search;_ylt=AwrXgyJDEwRagWkAZTvQtDMD;_ylu=X3oDMTEza3NiY3RnBGNvbG8DZ3ExBHBvcwMxBHZ0aWQDBHNlYwNwYWdpbmF0aW9u?p=Amazon&pz=10&fr=yfp-t-%5Bfpdmcntrl%2C+strm012%2C+FPNOT02%2C+SR202%5D&fr2=sb-top-news.search&bct=1&b=91&pz=10&bct=1&xargs=0']
	# download_delay = 5
	rules = (Rule(SgmlLinkExtractor(allow=(), restrict_xpaths=('//a[@class="next fc-14th"]',)), callback="parse_items", follow=True),)

	def parse_items(self, response):
		items = []
		sel = scrapy.selector.Selector(response)
		sites = sel.xpath('//div[@class="compTitle"]')
		for site in sites:
			item = YhscraperItem()
			item['title'] = site.xpath('h3/a//text()').extract()
			# item['link'] = site.xpath('h3/a/@href').extract()
			item['date'] = site.xpath('div/span[@class="tri fc-2nd ml-10"]//text()').extract()
			items.append(item)
		with open('./dataset/Amazon_news_title.csv','ab') as f:
			writer = csv.writer(f)
			today = datetime.datetime.now()
			for item in items:
				# Title processing
				title = ""
				for phrase in item['title']:
					title += phrase
				title = title.replace('\xc2\xa0', ' ')
				# Date processing
				date_list = item['date'][0].split(' ')
				date = ""				
				day = today.day
				month = today.month
				year = today.year
				if date_list[1] in ['month', 'year', 'months', 'years']:
					continue
				elif date_list[1] in ['hour', 'hours', 'minute','minutes', 'second', 'seconds']:
					if day < 10:
						day = '0' + str(day)
					if month < 10:
						month = '0' + str(month)
					date = "%s%s%s" % (year, month, day)
				else:
					N = int(date_list[0])
					date_N_days_ago = today - datetime.timedelta(days=N)
					year = date_N_days_ago.year
					month = date_N_days_ago.month
					day = date_N_days_ago.day
					if day < 10:
						day = '0' + str(day)
					if month < 10:
						month = '0' + str(month)
					date = "%s%s%s" % (year, month, day)
				# Write title and date into file
				writer.writerow([date, title])
				# f.write(date + ': ')
				# # f.write(item['link'][0] + '\r\n')
				# f.write(title +'\r\n')
		return

