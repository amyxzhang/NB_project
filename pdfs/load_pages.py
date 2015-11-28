from selenium import webdriver
import os
import time
# initiate the browser. It will open the url, 
# and we can access all its content, and make actions on it. 

chromedriver = "/usr/local/bin/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
browser = webdriver.Chrome(chromedriver)
browser.set_window_size(1350,1100)
'''
            15463, 
            18000,
            17423,
            15855,
            18565,
            15461,
            15460,
            15779,
            18887,
            15456,
'''

for i in range(1,120):
    url = 'http://127.0.0.1:8000/static/nbapp/15456.split/html/15456.%s.html' % i

    browser.get(url)
    time.sleep(.5)
