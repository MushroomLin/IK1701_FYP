import csv

DILIMITER = '(English)\n'
date_dict = {
    'January': '01',
    'February': '02',
    'March': '03',
    'April': '04',
    'May': '05',
    'June': '06',
    'July': '07',
    'August': '08',
    'September': '09',
    'October': '10',
    'November': '11',
    'December': '12'
}
FILE_NAME = 'Apple_News_Data.csv'
TOTLE_NUM_OF_FILE = 264

def write_one_news(cursor1, cursor2, csvwriter, content):
    # Find the title 
    title_cursor1 = cursor1 - 2
    while content[title_cursor1] != '\n' and content[title_cursor1] != ' \n':
        title_cursor1 -= 1
    title_cursor1 += 1
    title_cursor2 = cursor1 - 1
    title = ''
    for i in range(title_cursor1, title_cursor2+1):
        if i == title_cursor1:
            title += content[i][2:-1]
        else:
            title += content[i][:-1]

    # Find the date
    raw_date = content[cursor1].split(',')[-3]
    dmy = raw_date.split(' ')
    if len(dmy[-3]) == 1:
        date = dmy[-1]+'-'+date_dict[dmy[-2]]+'-'+'0'+dmy[-3]
    else:
        date = dmy[-1]+'-'+date_dict[dmy[-2]]+'-'+dmy[-3]

    # Find the leading paragraphs
    lp_cursor1 = cursor1 + 1
    lp_cursor2 = cursor2
    while content[lp_cursor2] != '\n' and content[lp_cursor2] != ' \n':
        lp_cursor2 -= 1
    lp_cursor2 -= 1
    lp = ''
    for i in range(lp_cursor1, lp_cursor2+1):
        lp += content[i][:-1]
    
    # Write one news
    csvwriter.writerow([date, title, lp])

# Main Control Flow
csvfile = open(FILE_NAME, 'ab')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(['Date', 'Title', 'Leading Paragraphs'])

file_num = 1
while file_num <= TOTLE_NUM_OF_FILE:
    # Read one file
    infile = open('./Apple_data_txt/'+str(file_num)+'.txt', 'r')
    content = infile.readlines()

    # Find initial cursor1
    cursor1 = 0
    while len(content[cursor1]) == 0 or content[cursor1].split(' ')[-1] != DILIMITER:
        cursor1 += 1 

    # Find initial cursor2
    cursor2 = cursor1 + 1
    while len(content[cursor2]) == 0 or content[cursor2].split(' ')[-1] != DILIMITER:
        cursor2 += 1 

    while True:
        write_one_news(cursor1,cursor2,csvwriter, content)
        # Update Cursors
        cursor1 = cursor2
        cursor2 += 1
        while cursor2 < len(content) and (len(content[cursor2]) == 0 or content[cursor2].split(' ')[-1] != DILIMITER):
            cursor2 += 1 
        if cursor2 == len(content):
            cursor2 = len(content) - 1
            write_one_news(cursor1,cursor2,csvwriter, content)
            break
    file_num += 1

csvfile.close()