import scrapy

import sys
import datetime
import calendar
import csv

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector

from yhscraper.items import YhscraperItem

reload(sys)
sys.setdefaultencoding('utf-8')

# Load command line parameters
init_url = sys.argv[2][6:]
nextpage_pattern = sys.argv[4][6:]
block_pattern = sys.argv[6][6:]
title_pattern = sys.argv[8][6:]
date_pattern = sys.argv[10][6:]
FILE_NAME = sys.argv[12][6:]

class YhSpider(CrawlSpider):
	name = "google" # Unique, name of the spider
	allowed_domains = ['finance.google.com']
	start_urls = [init_url]
	# download_delay = 10 # Uncomment this line if the program stops immediately
	rules = (Rule(SgmlLinkExtractor(allow=(), restrict_xpaths=(nextpage_pattern,)), callback="parse_items", follow=True),)

	def parse_items(self, response):
		items = []
		sel = scrapy.selector.Selector(response)
		sites = sel.xpath(block_pattern)
		for site in sites:
			item = YhscraperItem()
			item['title'] = site.xpath(title_pattern).extract()
			item['date'] = site.xpath(date_pattern).extract()
			items.append(item)
		with open('./dataset/'+FILE_NAME+'.csv','ab') as f:
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
				elif date_list[1] in ['day', 'days']:
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
				else:
					year = date_list[2]
					month = str(list(calendar.month_abbr).index(date_list[0]))
					day = date_list[1][:-1]
					if len(day) == 1:
						day = '0' + day
					if len(month) == 1:
						month = '0' + month
					date = "%s%s%s" % (year, month, day)
				# Write title and date into file
				writer.writerow([date, title])
		return

