from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

driver = webdriver.Chrome()

d = {
"products":[],
"prices":[],
"ratings":[],
"price_origs" : [],
"pages" : [],
}
 #List to store name of the product
 #List to store price of the product
 #List to store rating of the product

for i in range(1,7):

    driver.get("https://www.flipkart.com/laptops/pr?sid=6bo%2Cb5g&marketplace=FLIPKART&page=" + str(i))


    html  = driver.page_source
    html_soup = BeautifulSoup(html ,'html.parser')

    listed = html_soup.find_all("a",href=True, attrs={"class":"_1fQZEK"})

    for a in listed:

        name        =a.find('div', attrs={'class':'_4rR01T'})
        price       =a.find('div', attrs={'class':'_30jeq3 _1_WHN1'})
        price_orig  =a.find('div', attrs={'class':'_3I9_wc _27UcVY'})
        rating      =a.find('div', attrs={'class':'_3LWZlK'})

        if price_orig is None:
           price_orig = price 

        if rating is not None:
            rating =  rating.text

        d["products"].append(name.text)
        d["prices"].append(price.text)
        d["price_origs"].append(price_orig.text)
        d["ratings"].append(rating) 
        d["pages"].append(a['href'])

df = pd.DataFrame(d) 
df.to_csv('products.csv', index=False, encoding='utf-8')
