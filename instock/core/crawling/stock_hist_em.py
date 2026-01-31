#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Date: 2022/6/19 15:26
Desc: 东方财富网-行情首页-沪深京 A 股
"""
import random
import time

import pandas as pd
import math
from functools import lru_cache
from instock.core.eastmoney_fetcher import eastmoney_fetcher

__author__ = 'myh '
__date__ = '2025/12/31 '

# 创建全局实例，供所有函数使用
fetcher = eastmoney_fetcher()

def stock_zh_a_spot_em() -> pd.DataFrame:
    """
    东方财富网-沪深京 A 股-实时行情
    https://quote.eastmoney.com/center/gridlist.html#hs_a_board
    :return: 实时行情
    :rtype: pandas.DataFrame
    """
    url = "http://82.push2.eastmoney.com/api/qt/clist/get"
    page_size = 50
    page_current = 1
    params = {
        "pn": page_current,
        "pz": page_size,
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f15,f16,f17,f18,f20,f21,f22,f23,f24,f25,f26,f37,f38,f39,f40,f41,f45,f46,f48,f49,f57,f61,f100,f112,f113,f114,f115,f221",
        "_": "1623833739532",
    }
    r =  fetcher.make_request(url, params=params)
    data_json = r.json()
    data = data_json["data"]["diff"]
    if not data:
        return pd.DataFrame()

    data_count = data_json["data"]["total"]
    page_count = math.ceil(data_count/page_size)
    while page_count > 1:
        # 添加随机延迟，避免爬取过快
        time.sleep(random.uniform(0.5, 1.5))
        page_current = page_current + 1
        params["pn"] = page_current
        r =  fetcher.make_request(url, params=params)
        data_json = r.json()
        _data = data_json["data"]["diff"]
        data.extend(_data)
        page_count =page_count - 1

    temp_df = pd.DataFrame(data)
    temp_df.columns = [
        "最新价",
        "涨跌幅",
        "涨跌额",
        "成交量",
        "成交额",
        "振幅",
        "换手率",
        "市盈率动",
        "量比",
        "5分钟涨跌",
        "代码",
        "名称",
        "最高",
        "最低",
        "今开",
        "昨收",
        "总市值",
        "流通市值",
        "涨速",
        "市净率",
        "60日涨跌幅",
        "年初至今涨跌幅",
        "上市时间",
        "加权净资产收益率",
        "总股本",
        "已流通股份",
        "营业收入",
        "营业收入同比增长",
        "归属净利润",
        "归属净利润同比增长",
        "每股未分配利润",
        "毛利率",
        "资产负债率",
        "每股公积金",
        "所处行业",
        "每股收益",
        "每股净资产",
        "市盈率静",
        "市盈率TTM",
        "报告期"
    ]
    temp_df = temp_df[
        [
            "代码",
            "名称",
            "最新价",
            "涨跌幅",
            "涨跌额",
            "成交量",
            "成交额",
            "振幅",
            "换手率",
            "量比",
            "今开",
            "最高",
            "最低",
            "昨收",
            "涨速",
            "5分钟涨跌",
            "60日涨跌幅",
            "年初至今涨跌幅",
            "市盈率动",
            "市盈率TTM",
            "市盈率静",
            "市净率",
            "每股收益",
            "每股净资产",
            "每股公积金",
            "每股未分配利润",
            "加权净资产收益率",
            "毛利率",
            "资产负债率",
            "营业收入",
            "营业收入同比增长",
            "归属净利润",
            "归属净利润同比增长",
            "报告期",
            "总股本",
            "已流通股份",
            "总市值",
            "流通市值",
            "所处行业",
            "上市时间"
        ]
    ]
    temp_df["最新价"] = pd.to_numeric(temp_df["最新价"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
    temp_df["量比"] = pd.to_numeric(temp_df["量比"], errors="coerce")
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
    temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
    temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
    temp_df["今开"] = pd.to_numeric(temp_df["今开"], errors="coerce")
    temp_df["昨收"] = pd.to_numeric(temp_df["昨收"], errors="coerce")
    temp_df["涨速"] = pd.to_numeric(temp_df["涨速"], errors="coerce")
    temp_df["5分钟涨跌"] = pd.to_numeric(temp_df["5分钟涨跌"], errors="coerce")
    temp_df["60日涨跌幅"] = pd.to_numeric(temp_df["60日涨跌幅"], errors="coerce")
    temp_df["年初至今涨跌幅"] = pd.to_numeric(temp_df["年初至今涨跌幅"], errors="coerce")
    temp_df["市盈率动"] = pd.to_numeric(temp_df["市盈率动"], errors="coerce")
    temp_df["市盈率TTM"] = pd.to_numeric(temp_df["市盈率TTM"], errors="coerce")
    temp_df["市盈率静"] = pd.to_numeric(temp_df["市盈率静"], errors="coerce")
    temp_df["市净率"] = pd.to_numeric(temp_df["市净率"], errors="coerce")
    temp_df["每股收益"] = pd.to_numeric(temp_df["每股收益"], errors="coerce")
    temp_df["每股净资产"] = pd.to_numeric(temp_df["每股净资产"], errors="coerce")
    temp_df["每股公积金"] = pd.to_numeric(temp_df["每股公积金"], errors="coerce")
    temp_df["每股未分配利润"] = pd.to_numeric(temp_df["每股未分配利润"], errors="coerce")
    temp_df["加权净资产收益率"] = pd.to_numeric(temp_df["加权净资产收益率"], errors="coerce")
    temp_df["毛利率"] = pd.to_numeric(temp_df["毛利率"], errors="coerce")
    temp_df["资产负债率"] = pd.to_numeric(temp_df["资产负债率"], errors="coerce")
    temp_df["营业收入"] = pd.to_numeric(temp_df["营业收入"], errors="coerce")
    temp_df["营业收入同比增长"] = pd.to_numeric(temp_df["营业收入同比增长"], errors="coerce")
    temp_df["归属净利润"] = pd.to_numeric(temp_df["归属净利润"], errors="coerce")
    temp_df["归属净利润同比增长"] = pd.to_numeric(temp_df["归属净利润同比增长"], errors="coerce")
    temp_df["报告期"] = pd.to_datetime(temp_df["报告期"], format='%Y%m%d', errors="coerce")
    temp_df["总股本"] = pd.to_numeric(temp_df["总股本"], errors="coerce")
    temp_df["已流通股份"] = pd.to_numeric(temp_df["已流通股份"], errors="coerce")
    temp_df["总市值"] = pd.to_numeric(temp_df["总市值"], errors="coerce")
    temp_df["流通市值"] = pd.to_numeric(temp_df["流通市值"], errors="coerce")
    temp_df["上市时间"] = pd.to_datetime(temp_df["上市时间"], format='%Y%m%d', errors="coerce")

    return temp_df


@lru_cache()
def code_id_map_em() -> dict:
    """
    东方财富-股票和市场代码
    http://quote.eastmoney.com/center/gridlist.html#hs_a_board
    :return: 股票和市场代码
    :rtype: dict
    """
    url = "http://80.push2.eastmoney.com/api/qt/clist/get"
    page_size = 50
    page_current = 1
    params = {
        "pn": page_current,
        "pz": page_size,
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:1 t:2,m:1 t:23",
        "fields": "f12",
        "_": "1623833739532",
    }
    r =  fetcher.make_request(url, params=params)
    data_json = r.json()
    data = data_json["data"]["diff"]
    if not data:
        return dict()

    data_count = data_json["data"]["total"]
    page_count = math.ceil(data_count/page_size)
    while page_count > 1:
        # 添加随机延迟，避免爬取过快
        time.sleep(random.uniform(0.5, 1.5))
        page_current = page_current + 1
        params["pn"] = page_current
        r =  fetcher.make_request(url, params=params)
        data_json = r.json()
        _data = data_json["data"]["diff"]
        data.extend(_data)
        page_count =page_count - 1

    temp_df = pd.DataFrame(data)
    temp_df["market_id"] = 1
    temp_df.columns = ["sh_code", "sh_id"]
    code_id_dict = dict(zip(temp_df["sh_code"], temp_df["sh_id"]))
    page_current = 1
    params = {
        "pn": page_current,
        "pz": page_size,
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80",
        "fields": "f12",
        "_": "1623833739532",
    }
    r =  fetcher.make_request(url, params=params)
    data_json = r.json()
    data = data_json["data"]["diff"]
    if not data:
        return dict()

    data_count = data_json["data"]["total"]
    page_count = math.ceil(data_count/page_size)
    while page_count > 1:
        # 添加随机延迟，避免爬取过快
        time.sleep(random.uniform(0.5, 1.5))
        page_current = page_current + 1
        params["pn"] = page_current
        r =  fetcher.make_request(url, params=params)
        data_json = r.json()
        _data = data_json["data"]["diff"]
        data.extend(_data)
        page_count =page_count - 1

    temp_df_sz = pd.DataFrame(data)
    temp_df_sz["sz_id"] = 0
    code_id_dict.update(dict(zip(temp_df_sz["f12"], temp_df_sz["sz_id"])))
    page_current = 1
    params = {
        "pn": page_current,
        "pz": page_size,
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:81 s:2048",
        "fields": "f12",
        "_": "1623833739532",
    }
    r =  fetcher.make_request(url, params=params)
    data_json = r.json()
    data = data_json["data"]["diff"]
    if not data:
        return dict()

    data_count = data_json["data"]["total"]
    page_count = math.ceil(data_count/page_size)
    while page_count > 1:
        # 添加随机延迟，避免爬取过快
        time.sleep(random.uniform(0.5, 1.5))
        page_current = page_current + 1
        params["pn"] = page_current
        r =  fetcher.make_request(url, params=params)
        data_json = r.json()
        _data = data_json["data"]["diff"]
        data.extend(_data)
        page_count =page_count - 1

    temp_df_sz = pd.DataFrame(data)
    temp_df_sz["bj_id"] = 0
    code_id_dict.update(dict(zip(temp_df_sz["f12"], temp_df_sz["bj_id"])))
    return code_id_dict


def stock_zh_a_hist(
    symbol: str = "000001",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "20500101",
    adjust: str = "",
) -> pd.DataFrame:
    """
    东方财富网-行情首页-沪深京 A 股-每日行情
    https://quote.eastmoney.com/concept/sh603777.html?from=classic
    :param symbol: 股票代码
    :type symbol: str
    :param period: choice of {'daily', 'weekly', 'monthly'}
    :type period: str
    :param start_date: 开始日期
    :type start_date: str
    :param end_date: 结束日期
    :type end_date: str
    :param adjust: choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}
    :type adjust: str
    :return: 每日行情
    :rtype: pandas.DataFrame
    """
    code_id_dict = code_id_map_em()
    adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_dict[period],
        "fqt": adjust_dict[adjust],
        "secid": f"{code_id_dict[symbol]}.{symbol}",
        "beg": start_date,
        "end": end_date,
        "_": "1623766962675",
    }
    r =  fetcher.make_request(url, params=params)
    data_json = r.json()
    if not (data_json["data"] and data_json["data"]["klines"]):
        return pd.DataFrame()
    temp_df = pd.DataFrame(
        [item.split(",") for item in data_json["data"]["klines"]]
    )
    temp_df.columns = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
    ]
    temp_df.index = pd.to_datetime(temp_df["日期"])
    temp_df.reset_index(inplace=True, drop=True)

    temp_df["开盘"] = pd.to_numeric(temp_df["开盘"])
    temp_df["收盘"] = pd.to_numeric(temp_df["收盘"])
    temp_df["最高"] = pd.to_numeric(temp_df["最高"])
    temp_df["最低"] = pd.to_numeric(temp_df["最低"])
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"])
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"])
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"])
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"])
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"])
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"])

    return temp_df


def stock_zh_a_hist_min_em(
    symbol: str = "000001",
    start_date: str = "1979-09-01 09:32:00",
    end_date: str = "2222-01-01 09:32:00",
    period: str = "5",
    adjust: str = "",
) -> pd.DataFrame:
    """
    东方财富网-行情首页-沪深京 A 股-每日分时行情
    https://quote.eastmoney.com/concept/sh603777.html?from=classic
    :param symbol: 股票代码
    :type symbol: str
    :param start_date: 开始日期
    :type start_date: str
    :param end_date: 结束日期
    :type end_date: str
    :param period: choice of {'1', '5', '15', '30', '60'}
    :type period: str
    :param adjust: choice of {'', 'qfq', 'hfq'}
    :type adjust: str
    :return: 每日分时行情
    :rtype: pandas.DataFrame
    """
    code_id_dict = code_id_map_em()
    adjust_map = {
        "": "0",
        "qfq": "1",
        "hfq": "2",
    }
    if period == "1":
        url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "ndays": "5",
            "iscr": "0",
            "secid": f"{code_id_dict[symbol]}.{symbol}",
            "_": "1623766962675",
        }
        r =  fetcher.make_request(url, params=params)
        data_json = r.json()
        temp_df = pd.DataFrame(
            [item.split(",") for item in data_json["data"]["trends"]]
        )
        temp_df.columns = [
            "时间",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "成交量",
            "成交额",
            "最新价",
        ]
        temp_df.index = pd.to_datetime(temp_df["时间"])
        temp_df = temp_df[start_date:end_date]
        temp_df.reset_index(drop=True, inplace=True)
        temp_df["开盘"] = pd.to_numeric(temp_df["开盘"])
        temp_df["收盘"] = pd.to_numeric(temp_df["收盘"])
        temp_df["最高"] = pd.to_numeric(temp_df["最高"])
        temp_df["最低"] = pd.to_numeric(temp_df["最低"])
        temp_df["成交量"] = pd.to_numeric(temp_df["成交量"])
        temp_df["成交额"] = pd.to_numeric(temp_df["成交额"])
        temp_df["最新价"] = pd.to_numeric(temp_df["最新价"])
        temp_df["时间"] = pd.to_datetime(temp_df["时间"]).astype(str)
        return temp_df
    else:
        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": period,
            "fqt": adjust_map[adjust],
            "secid": f"{code_id_dict[symbol]}.{symbol}",
            "beg": "0",
            "end": "20500000",
            "_": "1630930917857",
        }
        r =  fetcher.make_request(url, params=params)
        data_json = r.json()
        temp_df = pd.DataFrame(
            [item.split(",") for item in data_json["data"]["klines"]]
        )
        temp_df.columns = [
            "时间",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "成交量",
            "成交额",
            "振幅",
            "涨跌幅",
            "涨跌额",
            "换手率",
        ]
        temp_df.index = pd.to_datetime(temp_df["时间"])
        temp_df = temp_df[start_date:end_date]
        temp_df.reset_index(drop=True, inplace=True)
        temp_df["开盘"] = pd.to_numeric(temp_df["开盘"])
        temp_df["收盘"] = pd.to_numeric(temp_df["收盘"])
        temp_df["最高"] = pd.to_numeric(temp_df["最高"])
        temp_df["最低"] = pd.to_numeric(temp_df["最低"])
        temp_df["成交量"] = pd.to_numeric(temp_df["成交量"])
        temp_df["成交额"] = pd.to_numeric(temp_df["成交额"])
        temp_df["振幅"] = pd.to_numeric(temp_df["振幅"])
        temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"])
        temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"])
        temp_df["换手率"] = pd.to_numeric(temp_df["换手率"])
        temp_df["时间"] = pd.to_datetime(temp_df["时间"]).astype(str)
        temp_df = temp_df[
            [
                "时间",
                "开盘",
                "收盘",
                "最高",
                "最低",
                "涨跌幅",
                "涨跌额",
                "成交量",
                "成交额",
                "振幅",
                "换手率",
            ]
        ]
        return temp_df


def stock_zh_a_hist_pre_min_em(
    symbol: str = "000001",
    start_time: str = "09:00:00",
    end_time: str = "15:50:00",
) -> pd.DataFrame:
    """
    东方财富网-行情首页-沪深京 A 股-每日分时行情包含盘前数据
    http://quote.eastmoney.com/concept/sh603777.html?from=classic
    :param symbol: 股票代码
    :type symbol: str
    :param start_time: 开始时间
    :type start_time: str
    :param end_time: 结束时间
    :type end_time: str
    :return: 每日分时行情包含盘前数据
    :rtype: pandas.DataFrame
    """
    code_id_dict = code_id_map_em()
    url = "https://push2.eastmoney.com/api/qt/stock/trends2/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "ndays": "1",
        "iscr": "1",
        "iscca": "0",
        "secid": f"{code_id_dict[symbol]}.{symbol}",
        "_": "1623766962675",
    }
    r =  fetcher.make_request(url, params=params)
    data_json = r.json()
    temp_df = pd.DataFrame(
        [item.split(",") for item in data_json["data"]["trends"]]
    )
    temp_df.columns = [
        "时间",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "最新价",
    ]
    temp_df.index = pd.to_datetime(temp_df["时间"])
    date_format = temp_df.index[0].date().isoformat()
    temp_df = temp_df[
        date_format + " " + start_time : date_format + " " + end_time
    ]
    temp_df.reset_index(drop=True, inplace=True)
    temp_df["开盘"] = pd.to_numeric(temp_df["开盘"])
    temp_df["收盘"] = pd.to_numeric(temp_df["收盘"])
    temp_df["最高"] = pd.to_numeric(temp_df["最高"])
    temp_df["最低"] = pd.to_numeric(temp_df["最低"])
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"])
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"])
    temp_df["最新价"] = pd.to_numeric(temp_df["最新价"])
    temp_df["时间"] = pd.to_datetime(temp_df["时间"]).astype(str)
    return temp_df


# if __name__ == "__main__":
#     stock_zh_a_spot_em_df = stock_zh_a_spot_em()
#     print(stock_zh_a_spot_em_df)
#
#     code_id_map_em_df = code_id_map_em()
#     print(code_id_map_em_df)
#
#     stock_zh_a_hist_df = stock_zh_a_hist(
#         symbol="000001",
#         period="daily",
#         start_date="20220516",
#         end_date="20220722",
#         adjust="hfq",
#     )
#     print(stock_zh_a_hist_df)
#
#     stock_zh_a_hist_min_em_df = stock_zh_a_hist_min_em(symbol="000001", period="1")
#     print(stock_zh_a_hist_min_em_df)
#
#     stock_zh_a_hist_pre_min_em_df = stock_zh_a_hist_pre_min_em(symbol="000001")
#     print(stock_zh_a_hist_pre_min_em_df)
#
#     stock_zh_a_spot_em_df = stock_zh_a_spot_em()
#     print(stock_zh_a_spot_em_df)
#
#     stock_zh_a_hist_min_em_df = stock_zh_a_hist_min_em(
#         symbol="000001", period='1'
#     )
#     print(stock_zh_a_hist_min_em_df)
#
#     stock_zh_a_hist_df = stock_zh_a_hist(
#         symbol="000001",
#         period="daily",
#         start_date="20170301",
#         end_date="20211115",
#         adjust="hfq",
#     )
#     print(stock_zh_a_hist_df)
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional
import logging
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection(db_url: str = "mysql+pymysql://root:wang521wei@192.168.31.192:3306/stock_master"):
    """
    获取数据库连接
    """
    try:
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise


def parse_symbol(symbol: str) -> tuple:
    """
    解析股票代码，处理带市场前缀的代码

    :param symbol: 股票代码，如 '000001', 'sh000001', 'sz000001', '600000', 'sh600000' 等
    :return: (market_prefix, clean_symbol)
    """
    symbol = symbol.lower().strip()

    # 定义市场前缀映射
    market_prefixes = {
        'sh': 'sh',  # 上海
        'sz': 'sz',  # 深圳
        'bj': 'bj',  # 北京
        'of': 'of',  # 基金
    }

    # 检查是否包含市场前缀
    for prefix in market_prefixes.keys():
        if symbol.startswith(prefix):
            # 提取市场前缀和股票代码
            market = symbol[:2]
            clean_code = symbol[2:]
            return market, clean_code

    # 如果没有市场前缀，根据代码判断市场
    # 6开头 - 上海，0/3开头 - 深圳，4/8开头 - 北京
    if symbol.startswith('6'):
        return 'sh', symbol
    elif symbol.startswith('0') or symbol.startswith('3'):
        return 'sz', symbol
    elif symbol.startswith('4') or symbol.startswith('8'):
        return 'bj', symbol
    else:
        # 默认返回原样
        return '', symbol


def convert_to_numeric(series):
    """
    安全地将Series转换为数值类型
    """
    try:
        # 先转换为字符串，然后替换可能的逗号分隔符
        series = series.astype(str).str.replace(',', '')
        # 转换为数值类型，错误值转为NaN
        return pd.to_numeric(series, errors='coerce')
    except Exception as e:
        logger.warning(f"转换数值类型失败: {e}")
        return series


def stock_zh_a_hist_from_db(
        symbol: str = "000001",
        period: str = "daily",
        start_date: str = "19700101",
        end_date: str = "20500101",
        adjust: str = "",
        db_url: Optional[str] = None
) -> pd.DataFrame:
    """
    从数据库读取沪深京 A 股历史行情数据
    支持多种股票代码格式:
    - '000001' (自动判断市场)
    - 'sh000001' (上证指数)
    - 'sz000001' (深证指数)
    - '600000' (浦发银行)
    - 'sh600000' (浦发银行，上海)
    - '000002' (万科A)
    - 'sz000002' (万科A，深圳)

    :param symbol: 股票代码
    :param period: 时间周期，暂时只支持'daily'，因为数据库存储的是日线数据
    :param start_date: 开始日期，格式: YYYYMMDD
    :param end_date: 结束日期，格式: YYYYMMDD
    :param adjust: 复权类型，数据库存储的是不复权数据，此参数仅保持接口兼容性
    :param db_url: 数据库连接URL，如果为None则使用默认配置
    :return: 每日行情DataFrame
    """

    # 暂时只支持日线数据，因为数据库表设计为日线
    if period != "daily":
        logger.warning(f"数据库版本暂时只支持日线数据(daily)，您选择了: {period}")
        return pd.DataFrame()

    # 如果提供了复权参数，给出警告（数据库存储的是不复权数据）
    if adjust != "":
        logger.warning("数据库存储的是不复权原始数据，adjust参数将被忽略")

    # 设置默认数据库连接
    if db_url is None:
        db_url = "mysql+pymysql://root:wang521wei@192.168.31.192:3306/stock_master"

    try:
        # 解析股票代码
        market_prefix, clean_symbol = parse_symbol(symbol)
        logger.info(f"解析股票代码: '{symbol}' -> 市场: '{market_prefix}', 代码: '{clean_symbol}'")

        # 转换日期格式
        start_date_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_date_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

        # 构建SQL查询
        sql = """
        SELECT 
            date as `日期`,
            CAST(opening_price AS DECIMAL(20,3)) as `开盘`,
            CAST(closing_price AS DECIMAL(20,3)) as `收盘`,
            CAST(highest_price AS DECIMAL(20,3)) as `最高`,
            CAST(lowest_price AS DECIMAL(20,3)) as `最低`,
            CAST(trading_volume AS UNSIGNED) as `成交量`,
            CAST(trading_value AS DECIMAL(20,2)) as `成交额`,
            CAST(rurnover_rate AS DECIMAL(20,2)) as `换手率`,
            CAST(pre_closing_price AS DECIMAL(20,3)) as `前收盘价`
        FROM daily_index 
        WHERE code = :symbol 
          AND date BETWEEN :start_date AND :end_date
        ORDER BY date ASC
        """

        # 获取数据库连接并执行查询
        engine = get_db_connection(db_url)

        with engine.connect() as conn:
            # 执行查询
            result = conn.execute(
                text(sql),
                {
                    "symbol": clean_symbol,  # 使用清理后的代码
                    "start_date": start_date_formatted,
                    "end_date": end_date_formatted
                }
            )
            # 如果查询结果为空，尝试带市场前缀的查询
            if market_prefix:
                logger.info(f"使用代码 '{clean_symbol}' 未找到数据，尝试带市场前缀查询")

                # 构建带市场前缀的查询
                sql_with_market = """
                SELECT 
                    date as `日期`,
                    CAST(opening_price AS DECIMAL(20,3)) as `开盘`,
                    CAST(closing_price AS DECIMAL(20,3)) as `收盘`,
                    CAST(highest_price AS DECIMAL(20,3)) as `最高`,
                    CAST(lowest_price AS DECIMAL(20,3)) as `最低`,
                    CAST(trading_volume AS UNSIGNED) as `成交量`,
                    CAST(trading_value AS DECIMAL(20,2)) as `成交额`,
                    CAST(rurnover_rate AS DECIMAL(20,2)) as `换手率`,
                    CAST(pre_closing_price AS DECIMAL(20,3)) as `前收盘价`
                FROM daily_index 
                WHERE code = :symbol 
                  AND date BETWEEN :start_date AND :end_date
                ORDER BY date ASC
                """

                full_symbol = f"{market_prefix}{clean_symbol}"
                result = conn.execute(
                    text(sql_with_market),
                    {
                        "symbol": full_symbol,
                        "start_date": start_date_formatted,
                        "end_date": end_date_formatted
                    }
                )

                temp_df = pd.DataFrame(result.fetchall(), columns=result.keys())

            # 如果查询结果为空，返回空DataFrame
            if temp_df.empty:
                logger.info(f"未找到股票 {symbol} 在 {start_date} 到 {end_date} 之间的数据")
                return pd.DataFrame()

            # 确保所有数值列都是正确的类型
            numeric_columns = ["开盘", "收盘", "最高", "最低", "成交量", "成交额",
                               "换手率", "前收盘价"]

            for col in numeric_columns:
                if col in temp_df.columns:
                    temp_df[col] = convert_to_numeric(temp_df[col])

            # 计算涨跌幅和涨跌额（基于前收盘价）
            # 安全计算，避免除零错误
            temp_df["涨跌幅"] = np.where(
                temp_df["前收盘价"] != 0,
                ((temp_df["收盘"] - temp_df["前收盘价"]) / temp_df["前收盘价"] * 100).round(4),
                0
            )

            temp_df["涨跌额"] = (temp_df["收盘"] - temp_df["前收盘价"]).round(4)

            # 计算振幅（基于前收盘价）
            temp_df["振幅"] = np.where(
                temp_df["前收盘价"] != 0,
                ((temp_df["最高"] - temp_df["最低"]) / temp_df["前收盘价"] * 100).round(4),
                0
            )

            # 设置索引
            temp_df["日期"] = pd.to_datetime(temp_df["日期"])
            temp_df.index = temp_df["日期"]
            temp_df.reset_index(inplace=True, drop=True)

            # 重新排列列顺序，与原始接口保持一致
            temp_df = temp_df[[
                "日期", "开盘", "收盘", "最高", "最低",
                "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"
            ]]

            # 最后再次确保数据类型正确
            for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额",
                        "振幅", "涨跌幅", "涨跌额", "换手率"]:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

            # 替换可能的NaN值
            temp_df = temp_df.fillna(0)

            logger.info(f"成功从数据库读取股票 {symbol} 的 {len(temp_df)} 条记录")

            return temp_df

    except Exception as e:
        logger.error(f"从数据库读取数据失败: {e}", exc_info=True)
        return pd.DataFrame()


def stock_zh_a_hist_from_db_all(
        symbol: str = "000001",
        period: str = "daily",
        start_date: str = "19700101",
        end_date: str = "20500101",
        adjust: str = "",
        db_url: Optional[str] = None
) -> pd.DataFrame:
    """
    从数据库读取沪深京 A 股历史行情数据
    支持多种股票代码格式:
    - '000001' (自动判断市场)
    - 'sh000001' (上证指数)
    - 'sz000001' (深证指数)
    - '600000' (浦发银行)
    - 'sh600000' (浦发银行，上海)
    - '000002' (万科A)
    - 'sz000002' (万科A，深圳)

    :param symbol: 股票代码
    :param period: 时间周期，暂时只支持'daily'，因为数据库存储的是日线数据
    :param start_date: 开始日期，格式: YYYYMMDD
    :param end_date: 结束日期，格式: YYYYMMDD
    :param adjust: 复权类型，数据库存储的是不复权数据，此参数仅保持接口兼容性
    :param db_url: 数据库连接URL，如果为None则使用默认配置
    :return: 每日行情DataFrame
    """

    # 暂时只支持日线数据，因为数据库表设计为日线
    if period != "daily":
        logger.warning(f"数据库版本暂时只支持日线数据(daily)，您选择了: {period}")
        return pd.DataFrame()

    # 如果提供了复权参数，给出警告（数据库存储的是不复权数据）
    if adjust != "":
        logger.warning("数据库存储的是不复权原始数据，adjust参数将被忽略")

    # 设置默认数据库连接
    if db_url is None:
        db_url = "mysql+pymysql://root:wang521wei@192.168.31.192:3306/stock_master"

    try:
        # 解析股票代码
        market_prefix, clean_symbol = parse_symbol(symbol)
        logger.info(f"解析股票代码: '{symbol}' -> 市场: '{market_prefix}', 代码: '{clean_symbol}'")

        # 转换日期格式
        start_date_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_date_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

        # 获取数据库连接并执行查询
        engine = get_db_connection(db_url)

        with engine.connect() as conn:
            # 执行查询

            # 如果查询结果为空，尝试带市场前缀的查询
            logger.info(f"使用代码 '{clean_symbol}' 未找到数据，尝试带市场前缀查询")

            # 构建带市场前缀的查询
            sql_with_market = """
            SELECT 
            di.date,
                si.code,
                si.name,
                di.date as `日期`,
                si.code as '代码',
                si.name as '名称',
                CAST(di.opening_price AS DECIMAL(20,3)) as `开盘`,
                CAST(di.closing_price AS DECIMAL(20,3)) as `收盘`,
                CAST(di.highest_price AS DECIMAL(20,3)) as `最高`,
                CAST(di.lowest_price AS DECIMAL(20,3)) as `最低`,
                CAST(di.trading_volume AS UNSIGNED) as `成交量`,
                CAST(di.trading_value AS DECIMAL(20,2)) as `成交额`,
                CAST(di.rurnover_rate AS DECIMAL(20,2)) as `换手率`,
                CAST(di.pre_closing_price AS DECIMAL(20,3)) as `前收盘价`
            FROM daily_index di ,stock_info si
            WHERE concat(si.exchange,si.code) = di.code and date BETWEEN :start_date AND :end_date
            ORDER BY date ASC
            """

            result = conn.execute(
                text(sql_with_market),
                {
                    "start_date": start_date_formatted,
                    "end_date": end_date_formatted
                }
            )

            temp_df = pd.DataFrame(result.fetchall(), columns=result.keys())

            # 如果查询结果为空，返回空DataFrame
            if temp_df.empty:
                logger.info(f"未找到股票 {symbol} 在 {start_date} 到 {end_date} 之间的数据")
                return pd.DataFrame()

            # 确保所有数值列都是正确的类型
            numeric_columns = ["开盘", "收盘", "最高", "最低", "成交量", "成交额",
                               "换手率", "前收盘价"]

            for col in numeric_columns:
                if col in temp_df.columns:
                    temp_df[col] = convert_to_numeric(temp_df[col])

            # 计算涨跌幅和涨跌额（基于前收盘价）
            # 安全计算，避免除零错误
            temp_df["涨跌幅"] = np.where(
                temp_df["前收盘价"] != 0,
                ((temp_df["收盘"] - temp_df["前收盘价"]) / temp_df["前收盘价"] * 100).round(4),
                0
            )

            temp_df["涨跌额"] = (temp_df["收盘"] - temp_df["前收盘价"]).round(4)

            # 计算振幅（基于前收盘价）
            temp_df["振幅"] = np.where(
                temp_df["前收盘价"] != 0,
                ((temp_df["最高"] - temp_df["最低"]) / temp_df["前收盘价"] * 100).round(4),
                0
            )

            # 设置索引
            temp_df["日期"] = pd.to_datetime(temp_df["日期"])
            temp_df.index = temp_df["日期"]
            temp_df.reset_index(inplace=True, drop=True)

            # 重新排列列顺序，与原始接口保持一致
            temp_df = temp_df[[
                "日期", "开盘", "收盘", "最高", "最低",
                "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率","代码","名称","code","name","date"
            ]]

            # 最后再次确保数据类型正确
            for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额",
                        "振幅", "涨跌幅", "涨跌额", "换手率"]:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

            # 替换可能的NaN值
            temp_df = temp_df.fillna(0)

            logger.info(f"成功从数据库读取股票 {symbol} 的 {len(temp_df)} 条记录")

            return temp_df

    except Exception as e:
        logger.error(f"从数据库读取数据失败: {e}", exc_info=True)
        return pd.DataFrame()


# 专门处理指数的函数
def stock_index_hist_from_db(
        symbol: str = "sh000001",
        period: str = "daily",
        start_date: str = "19700101",
        end_date: str = "20500101",
        db_url: Optional[str] = None
) -> pd.DataFrame:
    """
    专门处理指数数据的函数
    支持: sh000001(上证指数), sz399001(深证成指), sz399006(创业板指) 等
    """
    # 指数通常没有换手率，调整返回字段
    df = stock_zh_a_hist_from_db(
        symbol=symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
        db_url=db_url
    )

    if not df.empty:
        # 指数数据可能没有换手率字段，确保列存在
        if "换手率" not in df.columns:
            df["换手率"] = 0.0

        # 重新排序列
        expected_columns = [
            "日期", "开盘", "收盘", "最高", "最低",
            "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"
        ]

        # 确保所有列都存在
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0.0

        df = df[expected_columns]

    return df


# 批量查询函数
def batch_stock_hist_from_db(
        symbols: list,
        start_date: str,
        end_date: str,
        db_url: Optional[str] = None
) -> dict:
    """
    批量查询多只股票数据
    """
    results = {}
    for symbol in symbols:
        df = stock_zh_a_hist_from_db(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            db_url=db_url
        )
        results[symbol] = df
    return results


# 测试函数
def test_various_symbols():
    """测试不同格式的股票代码"""
    test_cases = [
        "000001",  # 上证指数（自动判断为sh）
        "sh000001",  # 上证指数（明确指定）
        "sz000001",  # 深证指数
        "399001",  # 深证成指（自动判断为sz）
        "sz399001",  # 深证成指
        "600000",  # 浦发银行（自动判断为sh）
        "sh600000",  # 浦发银行
        "000002",  # 万科A（自动判断为sz）
        "sz000002",  # 万科A
        "300750",  # 宁德时代（创业板，自动判断为sz）
        "sz300750",  # 宁德时代
    ]

    for symbol in test_cases:
        print(f"\n测试股票代码: {symbol}")
        df = stock_zh_a_hist_from_db(
            symbol=symbol,
            start_date="20240101",
            end_date="20240110"
        )

        if not df.empty:
            print(f"  找到 {len(df)} 条记录")
            print(f"  日期范围: {df['日期'].min()} 到 {df['日期'].max()}")
        else:
            print(f"  未找到数据")


# 主函数
if __name__ == "__main__":
    # 测试不同格式的股票代码
    print("=== 测试各种股票代码格式 ===")
    test_various_symbols()

    # 单独测试上证指数
    print("\n=== 测试上证指数 (sh000001) ===")
    df = stock_index_hist_from_db(
        symbol="sh000001",
        start_date="20240101",
        end_date="20240131"
    )

    if not df.empty:
        print(f"上证指数数据形状: {df.shape}")
        print(f"\n前5行数据:")
        print(df.head())
        print(f"\n数据统计:")
        print(df.describe())
    else:
        print("未找到上证指数数据")