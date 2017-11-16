from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as soup

#function that hops from page to page taking data and writing it to a .csv file
def letUsScrape(n):

    #telling webdriver to wait until it sees the class_name "item " on the page
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "item ")))

    #takes the html from the page an parses it
    page_html = driver.page_source
    #then we soup it
    #not neccessirly i just like using bs more
    page_soup = soup(page_html, 'lxml')
    #finding the items and looping to get what we need
    items = page_soup.find_all('div', class_="item")
    for item in items:

        picSite = (item.div.a.img["src"])
        itemIs = (item.div.a.img["alt"])
        itemPrice = (item.div.div.div.div.div.span.span.string)

        f.write(itemIs + "," + itemPrice + "," + picSite + "\n")
        #print(itemIs + "," + itemPrice + "," + picSite + "\n")


    p = str(n)
    url_next = 'https://shop.ccs.com/apparel?p=' + p
    driver.get(url_next)

#open the file and format it
filename = ("products.csv")
f = open(filename, "w")
headers = ("Product, Price, Picture\n")
f.write(headers)

#enter the name of the site you would like to scrape!
url = 'https://shop.ccs.com/'

#making an object of the webdriver
driver = webdriver.Firefox()
#telling the webdriver to go to the url
driver.get(url)
#telling the webdriver to wait until what i'm looking for is loaded on the apge
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "nav-container")))
#telling the webdriver to click the link on the page where the clothing items i am looking for are found
#can be more than one command depending on how many clicks it takes to get to where you're going
button = driver.find_element_by_link_text("Clothing")
button.click()

#s tells the webdriver what page you are going to start at
s = 2

for page in range(2, 59):

    letUsScrape(s)
    s = s + 1


f.close()

driver.quit()

