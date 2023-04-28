# -*- encoding: utf-8 -*-
from enum import Flag
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import time
import numpy as np
import requests
import json
import csv
import io

# 保存评论数据
def commentSave(list_comment):
    '''
    list_comment: 二维list,包含了多条用户评论信息
    '''
    file = io.open('/Users/xiaojun/Desktop/cv/huabaoxinneng/crawlering_tools/JDComment_Spider/data/Dxpower_JDComment_data_new.csv', 'w', encoding="utf-8", newline ='')
    writer = csv.writer(file)
    writer.writerow(['用户ID','评论内容','购买时间','点赞数','回复数','得分','评价时间','商品型号'])
    for i in range(len(list_comment)):
        writer.writerow(list_comment[i])
    file.close()
    print('存入成功')

def getCommentData(format_url,proc,i,maxPage):
    '''
    format_url: 格式化的字符串架子，在循环中给它添上参数
    proc: 商品的productID，标识唯一的商品号
    i: 商品的排序方式，例如全部商品、晒图、追评、好评等
    maxPage: 商品的评论最大页数
    '''
    sig_comment = []
    global list_comment
    cur_page = 0
    while cur_page < maxPage:
        cur_page += 1
        # url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv%s&score=%s&sortType=5&page=%s&pageSize=10&isShadowSku=0&fold=1'%(proc,i,cur_page)
        url = format_url.format(proc,i,cur_page) # 给字符串添上参数
        try:
            response = requests.get(url=url, headers=headers, verify=False)
            time.sleep(np.random.rand()*2)
            jsonData = response.text
            startLoc = jsonData.find('{')
            #print(jsonData[::-1])//字符串逆序
            jsonData = jsonData[startLoc:-2]
            jsonData = json.loads(jsonData)
            pageLen = len(jsonData['comments'])
            print("当前第%s页"%cur_page)
            for j in range(0,pageLen):
                userId = jsonData['comments'][j]['id']#用户ID
                content = jsonData['comments'][j]['content']#评论内容
                boughtTime = jsonData['comments'][j]['referenceTime']#购买时间
                voteCount = jsonData['comments'][j]['usefulVoteCount']#点赞数
                replyCount = jsonData['comments'][j]['replyCount']#回复数目
                starStep = jsonData['comments'][j]['score']#得分
                creationTime = jsonData['comments'][j]['creationTime']#评价时间
                referenceName = jsonData['comments'][j]['referenceName']#商品型号
                sig_comment.append(userId)#每一行数据
                sig_comment.append(content)
                sig_comment.append(boughtTime)
                sig_comment.append(voteCount)
                sig_comment.append(replyCount)
                sig_comment.append(starStep)
                sig_comment.append(creationTime)
                sig_comment.append(referenceName)
                list_comment.append(sig_comment)
                print(sig_comment)
                sig_comment = []
            if cur_page == maxPage:
                # 保存数据
                commentSave(list_comment)
                return
        except:
            time.sleep(5)
            cur_page -= 1
            print('网络故障或者是网页出现了问题，五秒后重新连接')

if __name__ == "__main__":
    global list_comment
    ua=UserAgent()
    format_url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&{0}&score={1}&sortType=5&page={2}&pageSize=10&isShadowSku=0&fold=1'
    # 设置访问请求头
    headers = {
    'Accept': '*/*',
    'Host':"club.jd.com",
    "User-Agent":ua.random,
    'Referer':"https://item.jd.com/",
    'sec-ch-ua':"\"Chromium\";v=\"92\", \" Not A;Brand\";v=\"99\", \"Google Chrome\";v=\"92\"",
    'sec-ch-ua-mobile': '?0',
    'Sec-Fetch-Dest': 'script',
    'Sec-Fetch-Mode':'no-cors',
    'Sec-Fetch-Site':'same-site',
    # 'cookie':'your cookies'
    'cookie':'__jdu=16673749640401560447357; shshshfpa=6ac3c07e-ef74-1189-db51-2aad7086f5c9-1643634445; cud=fa836c31f4e672e679dc7c8423627025; shshshfpx=6ac3c07e-ef74-1189-db51-2aad7086f5c9-1643634445; pinId=WJtyGqltHuIJYxXKg9-YbA; TrackID=1LNHLJHgedIjPnKRok293Seh73YaHjrJmlybEXTf6FvFpVZ6qrjGvB_-vbNsW0EQrwPxf_P6UmayHwERn97bgEHr3zJCWbUW8BY4GUTz0QZ407lVoUMBK5X51R35rxd9s; pin=jessiewei7; unick=jessiewei7; _tp=5dmb7izy9RwYjXqmD1TisA==; _pst=jessiewei7; shshshfp=012256d87673035f3e41d1b76f093edf; __jdv=122270672|www.hello-tech.com|-|referral|-|1681551607204; areaId=16; PCSYCityID=CN_350000_350400_0; jsavif=1; __jdc=122270672; __jda=122270672.16673749640401560447357.1667374964.1681786975.1681789482.20; cvt=12; 3AB9D23F7A4B3C9B=ZLQLCJQTLKVCSGBVT6JS3AWELQWB7OGG4N3OLZA3ECA4RAJENK7BWOT6CZC5OME7GPVONNHTGVQ4MOS73JXUQ6KWUE; RT="z=1&dm=jd.com&si=3r6iakjyrk&ss=lglojzgl&sl=0&tt=0&nu=38834d5355dd94db260c61576b3dd8da&cl=1hcyv"; token=00d1d281f0f81e7fd896b42b23d6af5a,2,934327; __tk=AdItR0b6k0kKXYXclD40hsbymEo0SuRbAbktSsb1jvg0jDW1guAPnzhyWvhQYYb6AbXphYMd,2,934327; ipLoc-djd=16-1317-1322-22758; 3AB9D23F7A4B3CSS=jdd03ZLQLCJQTLKVCSGBVT6JS3AWELQWB7OGG4N3OLZA3ECA4RAJENK7BWOT6CZC5OME7GPVONNHTGVQ4MOS73JXUQ6KWUEAAAAMHSKAUB6IAAAAACEBYRC6IUXUB64X; shshshsID=ec6fc88c6f38c2c472bb16e9e1083138_8_1681790173639; __jdb=122270672.8.16673749640401560447357|20.1681789482; shshshfpb=y8LO2trejbYBd+oU09YeUcQ==; csn=8'
    }
    #产品id参数
    productid = ['productId=68105648246']
    # productid = ['productId=45528625958','productId=68105648246','productId=10062490657801','productId=10062490657802','productID=10063257077729', 'productID=10063257077730','productID=10071022465181', 'productID=10071022465182', 'productID=10071022465183']
    list_comment = [[]]
    sig_comment = []
    for proc in productid:#遍历产品种类
        i = -1
        while i < 7:#遍历排序方式
            i += 1
            if(i == 6):
                continue
             #先访问第0页获取最大页数，再进行循环遍历
            url = format_url.format(proc,i,0)
            print(url)
            try:
                response = requests.get(url=url, headers=headers, verify=False)
                jsonData = response.text
                startLoc = jsonData.find('{')
                jsonData = jsonData[startLoc:-2]
                jsonData = json.loads(jsonData)
                print("最大页数%s"%jsonData['maxPage']) # 100
                getCommentData(format_url,proc,i,jsonData['maxPage'])#遍历每一页
                # commentSave(list_comment)
            except Exception as e:
                i -= 1
                print("the error is ",e)
                print("wating---")
                time.sleep(5)
                commentSave(list_comment)
    print("爬取结束，开始存储-------")
    commentSave(list_comment)
