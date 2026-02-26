


# http://www.baostock.com/mainContent?file=stockKData.md

import requests
import pymysql
import baostock as bs

from datetime import datetime

from datetime import timedelta

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)


DB_CONFIG = {
    "host": "192.168.31.192",
    "user": "root",
    "password": "wang521wei",
    "database": "stock_master",
    "charset": "utf8mb4"
}


def padding(param):
  if len(param) < 5:
    return "0" * (5 - len(param)) + param
  else:
    return param


def fetch_data():
  conn1 = pymysql.connect(host="192.168.31.192", user="root", password="wang521wei", database="stock_master")
  cursor1 = conn1.cursor()

  cursor1.execute("select code,exchange from stock_info where exchange like 'sz' or exchange like 'sh'")
  stockinfo = cursor1.fetchall()

  for stock in stockinfo:
    exchange = stock[1]
    code = stock[0]
    rs = bs.query_history_k_data_plus(f"{exchange}.{code}",
                                      "date,date,code,open,high,low,close,volume,amount,adjustflag",
                                      start_date='2026-02-24', end_date='2026-02-25',
                                      frequency="d", adjustflag="1")
    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
      # 获取一条记录，将记录合并在一起
      inner_dict = rs.get_row_data()
      print(f"外层键: {code}")
      stock = f"{exchange}{code}"
      datetime = inner_dict[0]
      open = inner_dict[3]
      close = inner_dict[6]  #
      high = inner_dict[4]  #
      low = inner_dict[5]
      volume = inner_dict[7]
      amount = inner_dict[8]

      cursor1.execute("""
                            INSERT INTO daily_index (code, date, opening_price, highest_price, lowest_price,
                                                  closing_price, trading_volume, trading_value)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
  
                        """, (stock, datetime, open, high, low, close, volume, amount))

      conn1.commit()






def isBusinessDay():
  response = requests.get(f"http://192.168.31.192:8088/user/isBusinessDay")
  return response.text=="true"


if __name__ == "__main__":

  # if not isBusinessDay():
  #   print("不是工作日")
  #   sys.exit()

  fetch_data()
