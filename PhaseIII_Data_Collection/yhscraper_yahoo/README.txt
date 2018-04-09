Web crawler for Yahoo Finance.
The crawlers work in an iterative manner, with the link of the 'next-page' is provided by user.

Run the script:
$ scrapy crawl -a argv1=[argv1] -a argv2=[argv2] -a argv3=[argv3] -a argv4=[argv4] -a argv5=[argv5] -a argv6=[argv6] yh
argv1: url of the initial page
argv2: url of the 'next-page' pattern (e.g., //a[@class="next fc-14th"])
argv3: the html pattern of the block to scrape (e.g., //div[@class="compTitle"])
argv4: the html pattern of title (e.g., h3/a//text())
argv5: the html pattern of date (e.g., div/span[@class="tri fc-2nd ml-10"]//text())
argv6: the output filename ('.csv' will be automatically added as the postfix)
The output file containing date and title of news will be produced and stored in /yhscraper/dataset.