import  requests
from bs4 import BeautifulSoup

url = 'http://dm.postech.ac.kr/~cartopy/ConvMF/data/'
res = requests.get(url)
res = res.text.encode(res.encoding).decode('utf-8')
soup = BeautifulSoup(res, 'html.parser')
for li in soup.find_all(name='li'):
    li.find(name='a')['href']
