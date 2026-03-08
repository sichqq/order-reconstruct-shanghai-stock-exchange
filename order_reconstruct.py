# -*- coding: utf-8 -*-
#%% 导入所需要的包

import datetime
import time, os, sys, traceback
from pprint import pp
import numpy as np

import pandas as pd
import polars as pl
import numpy as np


import pickle, multiprocessing, warnings
from time import perf_counter   # python3.8专用
from datetime import datetime, timedelta




#忽略警告信息
warnings.filterwarnings("ignore")





# measure_time 是一个装饰器函数，它接受一个函数作为输入，并返回一个新的函数 wrapper
def measure_time(func):
    def wrapper(*args, **kwargs):

        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        execution_time = (end_time - start_time)  / 1e6
        print(f"函数 {func.__name__} 的执行时间为：",  "{:.6f} ms".format(execution_time))

        return result
    return wrapper






# 新建目录
def mkdir(path): 
    path=path.strip()    # 去除首位空格
    path=path.rstrip("\\")    # 去除尾部 \ 符号
    isExists=os.path.exists(path)
    if not isExists:    # 判断结果
        os.makedirs(path) 







# 将13位的时间戳 '1632360743570'  转换成str格式的时间  '2021-09-23 09:32:23.570'
def unix_to_strtime(unix):
    strtime = ''
    try:
        t_unix = str(unix)[:10]
        ms = str(unix)[10:]

        timeStamp = int(t_unix)
        m_time = time.localtime(timeStamp)
        # print('unix_to_strtime() m_time: ', m_time)
        strtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeStamp))
        strtime = strtime + '.' + ms

    except Exception as e:
        print('unix_to_strtime() Exception: ', e)
    return strtime






# 将13位的时间戳 '1632360743570'  转换成str格式的时间  "2025.03.27T09:14:00.790"
def unix_to_strtime_t(unix):
    strtime = ''
    try:
        t_unix = str(unix)[:10]
        ms = str(unix)[10:]

        timeStamp = int(t_unix)
        m_time = time.localtime(timeStamp)

        strtime = time.strftime("%Y.%m.%dT%H:%M:%S", time.localtime(timeStamp))
        strtime = strtime + '.' + ms

    except Exception as e:
        print('unix_to_strtime_t() Exception: ', e)
    return strtime






# 转换WT.parquet和CJ.parquet文件到entrust.csv和trade.csv格式
def convert_parquet_files(wt_file_path, cj_file_path, entrust_output_path, trade_output_path):
  
    # 公共列名
    # S_COLUMNS = ['TradeTime', 'LocalTime', 'BizIndex', 'ApplSeqNum', 'OfferApplSeqNum', 'BidApplSeqNum']
    S_COLUMNS = ['TradeTime', 'Unix', 'BizIndex', 'ApplSeqNum', 'OfferApplSeqNum', 'BidApplSeqNum']
      

    # entrust.csv专有列名
    WT_COLUMNS = ['OrderType', 'Side', 'Price', 'OrderQty']
    
    # trade.csv专有列名
    CJ_COLUMNS = ['TradeBSFlag', 'TradePrice', 'TradeQty']
    
    print("开始转换WT.parquet文件...")
    

    try:
        # 读取WT.csv文件
        # 注意：WT.csv文件使用'Code'作为股票代码列
        # wt_df = pd.read_csv(wt_file_path)
        wt_df = pd.read_parquet(wt_file_path, engine='pyarrow')
        print(f"WT.csv读取成功，共{len(wt_df)}行数据")
        
        # nCode,nTime,nOrder,nPrice,nVolume,nBroker,chOrderKind,chFunctionCode,unix
        # 688800,91400690,0,0,0,371,83,73,1770858840690
        # 688800,91502100,3010,770500,200,369685,65,66,1770858902100

        # 创建entrust.csv的DataFrame
        entrust_df = pd.DataFrame()
        

        # 将13位的时间戳 '1632360743570'  转换成str格式的时间  '2021-09-23 09:32:23.570'
        # entrust_df['TradeTime'] = wt_df['unix'].apply(unix_to_strtime)

        # 将13位的时间戳 '1632360743570'  转换成str格式的时间  "2025.03.27T09:14:00.790"
        entrust_df['TradeTime'] = wt_df['unix'].apply(unix_to_strtime_t)


        entrust_df['Unix'] = wt_df['unix']


        # 3. 转换BizIndex列（从Broker列）
        entrust_df['BizIndex'] = wt_df['nBroker']
        
        # 4. 转换ApplSeqNum列（从Order列）
        entrust_df['ApplSeqNum'] = wt_df['nOrder']
        
        # 5. 转换OfferApplSeqNum列（entrust.csv中通常为空，用NaN填充）
        entrust_df['OfferApplSeqNum'] = np.nan
        
        # 6. 转换BidApplSeqNum列（entrust.csv中通常为空，用NaN填充）
        entrust_df['BidApplSeqNum'] = np.nan
        
        # 7. 转换专有列
        
        # OrderType列（从OrderKind列）
        # WT.csv中OrderKind列值：D, A等
        # entrust_df['OrderType'] = wt_df['chOrderKind']
        entrust_df['OrderType'] = wt_df['chOrderKind'] .apply( 
            lambda x: chr(x) if pd.notnull(x) and x < 128 else str(x)
        )

        # Side列（从FunctionCode列）
        # WT.csv中FunctionCode列值：B, S等
        # B表示买入，S表示卖出
        # entrust_df['Side'] = wt_df['chFunctionCode']
        entrust_df['Side'] = wt_df['chFunctionCode'] .apply( 
            lambda x: chr(x) if pd.notnull(x) and x < 128 else str(x)
        )

        
        # Price列（从Price列，需要乘以0.0001）
        # 注意：原始Price列可能是整型，需要转换为浮点型并乘以0.0001
        entrust_df['Price'] = wt_df['nPrice'] * 0.0001

        
        # OrderQty列（从Volume列）
        entrust_df['OrderQty'] = wt_df['nVolume']
        
        # 重排列顺序：先公共列，再专有列
        entrust_df = entrust_df[S_COLUMNS + WT_COLUMNS]
        
        # 保存entrust.csv
        entrust_df.to_csv(entrust_output_path, index=False)
        print(f"entrust.csv保存成功，共{len(entrust_df)}行数据")
        
    except Exception as e:
        print(f"转换WT.parquet时出错: {e}")
        traceback.print_exc()
        return

    
    print("\n开始转换CJ.parquet文件...")
        # python order_rc.py


    try:
        # 读取CJ.csv文件
        # cj_df = pd.read_csv(cj_file_path)
        cj_df = pd.read_parquet(cj_file_path, engine='pyarrow')
        print(f"CJ.parquet读取成功，共{len(cj_df)}行数据")
        # print("cj_df.columns: \n", cj_df.columns)
   

        # nCode,nTime,nIndex,nPrice,nVolume,nBSFlag,chFunctionCode,nAskOrder,nBidOrder,unix
        # 688800,92500610,369947,800300,800,66,0,370587,390439,1770859500610
        # 688800,92500610,369948,800300,500,83,0,370587,354770,1770859500610

        # 创建trade.csv的DataFrame
        trade_df = pd.DataFrame()
        
        # 将13位的时间戳 '1632360743570'  转换成str格式的时间  '2021-09-23 09:32:23.570'
        # trade_df['TradeTime'] = cj_df['unix'].apply(unix_to_strtime)
       
        # 将13位的时间戳 '1632360743570'  转换成str格式的时间  "2025.03.27T09:14:00.790"
        trade_df['TradeTime'] = cj_df['unix'].apply(unix_to_strtime_t)


        trade_df['Unix'] = cj_df['unix']

        # 3. 转换BizIndex列（从nIndex列）
        trade_df['BizIndex'] = cj_df['nIndex']
        
        # 4. 转换ApplSeqNum列（trade.csv中通常为空，用NaN填充）
        trade_df['ApplSeqNum'] = np.nan
        
        # 5. 转换OfferApplSeqNum列（从nAskOrder列）
        trade_df['OfferApplSeqNum'] = cj_df['nAskOrder']
        
        # 6. 转换BidApplSeqNum列（从nBidOrder列）
        trade_df['BidApplSeqNum'] = cj_df['nBidOrder']
        
        # 7. 转换专有列
        
        # TradeBSFlag列（从nBSFlag列）
        # 示例中nBSFlag值为83, 66等，需要转换为字符类型
        trade_df['TradeBSFlag'] = cj_df['nBSFlag'].apply(
            lambda x: chr(x) if pd.notnull(x) and x < 128 else str(x)
        )
        
        # TradePrice列（从nPrice列，需要乘以0.0001）
        trade_df['TradePrice'] = cj_df['nPrice'] * 0.0001
        
        # TradeQty列（从nVolume列）
        trade_df['TradeQty'] = cj_df['nVolume']
        
        # 重排列顺序：先公共列，再专有列
        trade_df = trade_df[S_COLUMNS + CJ_COLUMNS]
        
        # 保存trade.csv
        trade_df.to_csv(trade_output_path, index=False)
        print(f"trade.csv保存成功，共{len(trade_df)}行数据")
        
        print("\n所有转换完成！")
        
    except Exception as e:
        print(f"转换CJ.parquet时出错: {e}")
        traceback.print_exc()






# 基于ApplSeqNum列进行分组聚合
def aggregate_by_applseqnum(df):
    """
    基于ApplSeqNum列进行分组聚合
    
    参数:
    -----------
    df : pandas.DataFrame
        输入数据框，包含列: [TradeTime, Unix, BizIndex, ApplSeqNum, OrderType, Side, Price, OrderQty]
    
    返回:
    --------
    pandas.DataFrame
        聚合后的数据框，OrderQty列求和，其他列取组内第一个值
    """
    
    # 定义聚合规则：OrderQty求和，其他列取第一个值
    agg_dict = {'OrderQty': 'sum'}
    
    # 其他列都取第一个值
    other_cols = [col for col in df.columns if col not in ['ApplSeqNum', 'OrderQty']]
    for col in other_cols:
        agg_dict[col] = 'first'
    
    # 按ApplSeqNum分组并聚合
    result = df.groupby('ApplSeqNum', as_index=False).agg(agg_dict)
    
    # 调整列顺序，保持与原始数据框一致
    result = result[df.columns]
    
    return result






# 重建上海证券交易所逐笔委托数据，添加TradedQty列记录即时成交数量
def reconstruct_entrust_data(entrust_path, trade_path, new_entrust_path, time_window_ms=100):
    try:
        """
        重建上海证券交易所逐笔委托数据，添加TradedQty列记录即时成交数量
        
        参数:
            entrust_path: 原始逐笔委托CSV文件路径
            trade_path: 逐笔成交CSV文件路径  
            new_entrust_path: 重建后的逐笔委托CSV文件路径
            time_window_ms: 时间窗口参数，单位为毫秒，默认为100毫秒
        """
        
        print("开始读取数据文件...")
        
        # 1. 读取CSV文件
        try:
            # 读取逐笔委托文件
            dtype = {"BizIndex":int, "ApplSeqNum":int}
            # TradeTime,LocalTime,BizIndex,ApplSeqNum,OfferApplSeqNum,BidApplSeqNum,OrderType,Side,Price,OrderQty
            # 2026.01.20T09:14:00.140,09:14:00.140,142,0,,,S,I,0.0,0
            # 2026.01.20T09:15:01.790,09:15:01.790,415703,5120,,,A,S,6.88,2000
            entrust_df = pd.read_csv(entrust_path, dtype=dtype)
            print(f"成功读取逐笔委托文件，共{len(entrust_df)}行")

            dtype = {"BizIndex":int, "OfferApplSeqNum":int, "BidApplSeqNum":int}
            # TradeTime,LocalTime,BizIndex,ApplSeqNum,OfferApplSeqNum,BidApplSeqNum,TradeBSFlag,TradePrice,TradeQty
            # 2026.01.20T09:25:01.240,09:25:01.240,416461,,439740,313833,S,6.75,2000
            # 读取逐笔成交文件
            trade_df = pd.read_csv(trade_path, dtype=dtype)
            print(f"成功读取逐笔成交文件，共{len(trade_df)}行")
            
        except Exception as e:
            print(f"读取文件失败: {e}")
            return
        


        # 转换逐笔委托时间
        # entrust_df['Unix'] = pd.to_datetime(entrust_df['TradeTime'], format='%Y.%m.%dT%H:%M:%S.%f', errors='coerce')
        entrust_df['type'] = 'entrust'
        entrust_df["ApplSeqNum"] = entrust_df["ApplSeqNum"].astype(int)


        # 转换逐笔成交时间
        # trade_df['Unix'] = pd.to_datetime(trade_df['TradeTime'], format='%Y.%m.%dT%H:%M:%S.%f', errors='coerce')
        trade_df['type'] = 'trade'
        trade_df["OfferApplSeqNum"] = trade_df["OfferApplSeqNum"].astype(int)
        trade_df["BidApplSeqNum"] = trade_df["BidApplSeqNum"].astype(int)


        trade_df = trade_df[["TradeTime", "Unix", "BizIndex", "BidApplSeqNum", "OfferApplSeqNum", 
        "TradePrice", "TradeQty", "type" ]]

        entrust_df = entrust_df[["TradeTime", "Unix", "BizIndex", "ApplSeqNum", "OrderType", 
        "Side", "Price", "OrderQty", "type" ]]


        df = pd.concat([entrust_df, trade_df])
        df.sort_values(by=['Unix', 'BizIndex'], inplace=True)
        df = df.reset_index(drop=True)
        df = df.fillna(0)

        df["OfferApplSeqNum"] = df["OfferApplSeqNum"].astype(int)
        df["BidApplSeqNum"] = df["BidApplSeqNum"].astype(int)
        df["ApplSeqNum"] = df["ApplSeqNum"].astype(int)


        df['idx'] = pd.Series(range(df.shape[0]), index=df.index)     # 将index复制为新的一列

        # Fill or Kill (FOK) - 立即全部成交，否则取消
        df['FOK'] = ""     # 标记所有的即时成交记录




        print("df.shape: ", df.shape)
        lst_new_order = []

 
        for idx in range(df.shape[0]):
            if idx % 100 == 0:
                print("df.iterrows(), idx: ", idx)

            try:
                if df['type'].iloc[idx] == 'trade':
                    trade_qty = df['TradeQty'].iloc[idx]
                    biz_index = df['BizIndex'].iloc[idx]
                    unix = df['Unix'].iloc[idx]
                    trade_time = df["TradeTime"].iloc[idx]
                                                
                    bid_order_no = int(df['BidApplSeqNum'].iloc[idx])            # 尝试查找买方订单
                    offer_order_no = int(df['OfferApplSeqNum'].iloc[idx])        # 尝试查找卖方订单

                    # trade_df = trade_df[["TradeTime", "Unix", "BizIndex", "BidApplSeqNum", "OfferApplSeqNum", 
                    # "TradePrice", "TradeQty", "type" ]]
                    TradePrice = df['TradePrice'].iloc[idx]
                    TradeQty = int(df['TradeQty'].iloc[idx])


                    df1 = df[ (df["idx"]<idx) & (df["type"]=="entrust") ]
                    if df1.shape[0] > 0:

                        # 获取在当前成交之前的所有委托编号表
                        order_list = df1["ApplSeqNum"].tolist()

                        # if bid_order_no in order_list or offer_order_no in order_list:
                        #     print(f"idx: {idx}, bid_order_no: {bid_order_no}, offer_order_no: {offer_order_no}")
                        # entrust_df = entrust_df[["TradeTime", "Unix", "BizIndex", "ApplSeqNum", "OrderType", 
                        # "Side", "Price", "OrderQty", "type" ]]

                        if bid_order_no not in order_list:
                            df['FOK'].iloc[idx] = "bid"     # 即时成交, 买方主动
                            # 此时买方的委托记录没有出现在委托列表中，需要建立一条新的主动买委托

                            # "OrderType": A & D,  "Side": B & S
                            dic_x = {"TradeTime":trade_time, "Unix":unix, "BizIndex":biz_index, 
                            "ApplSeqNum":bid_order_no, "OrderType":'A', "Side":'B',
                            "Price":TradePrice, "OrderQty":TradeQty}
                            lst_new_order.append(dic_x)


                        if offer_order_no not in order_list:
                            df['FOK'].iloc[idx] = "ask"     # 即时成交, 卖方主动
                            # 此时卖方的委托记录没有出现在委托列表中，需要建立一条新的主动卖委托
                            dic_x = {"TradeTime":trade_time, "Unix":unix, "BizIndex":biz_index, 
                            "ApplSeqNum":offer_order_no, "OrderType":'A', "Side":'S',
                            "Price":TradePrice, "OrderQty":TradeQty}
                            lst_new_order.append(dic_x)
 

            except Exception as e:
                print("reconstruct_entrust_data() Exception for: ", e)


        if lst_new_order:
            df2 = pd.DataFrame(lst_new_order)
            df2 = aggregate_by_applseqnum(df2)        # 基于ApplSeqNum列进行分组聚合
            # df2.to_csv(new_entrust_path, index=False)

            # TradeTime   Unix    BizIndex    ApplSeqNum  OrderType   Side    Price   OrderQty
            # 2026.01.20T09:30:02.260 30:02.3 580691  499805  A   B   6.75    3000
            # 2026.01.20T09:30:02.530 30:02.5 582567  501168  A   B   6.75    700

            df20 = df2[["ApplSeqNum", "OrderQty"]]
            df20.rename(columns={"OrderQty": "TradedQty"}, inplace=True)
            # entrust_df["ApplSeqNum"] = entrust_df["ApplSeqNum"].astype(int)

 
            # 左连接：保留左表所有行, 原始委托表 和 有痕迹的委托表 合并
            df3 = pd.merge(entrust_df, df20, on="ApplSeqNum", how='left')
            df3 = df3.fillna(0)


            # 原始委托和有痕迹委托合并表
            df3.sort_values(by=['Unix', 'BizIndex'], inplace=True)
            df3 = df3.reset_index(drop=True)

            lst_seqnum2 = df20["ApplSeqNum"].tolist()
            lst_seqnum3 = df3["ApplSeqNum"].tolist()

            # # 方法2b：纯set差集（不保证顺序，去重）
            lst_seqnum1 = list(set(lst_seqnum2) - set(lst_seqnum3))    # 获取 lst_seqnum2 中不在 lst_seqnum3 中的元素


            df5 = pd.DataFrame({"ApplSeqNum": lst_seqnum1})    # 纯打单的委托ID组成一个dataframe

           # 左连接：保留左表所有行, 获取 纯打单委托表
            df6 = pd.merge(df5, df2, on="ApplSeqNum", how='left')
            df6 = df6.fillna(0)
            df6["CDD"] = 1    # 纯打单标记


            # 将原始委托和有痕迹委托合并的表, 再与纯打单委托合并, 纵向拼接（行增加）
            df7 = pd.concat([df3, df6], axis=0)

            df7 = df7.sort_values(by=['BizIndex'], ascending=[True])
            df7 = df7.reset_index(drop=True)

            df7.fillna(0, inplace=True)
            df7["OrderQty_F"] = df7["OrderQty"] + df7["TradedQty"]

            df7 = df7.drop(columns=['type'])


            # 即时成交余量无
            # 即时成交余量委托
            # 先委托后成交
            # 计算上面三种委托类型的apply函数
            def apply_type(x):
                types = "先委托后成交"
                if x["CDD"] > 0:
                    types = "即时成交余量无"
                elif x["TradedQty"] > 0:
                    types = "即时成交余量委托"
                return types


            df7["类型"] = df7.apply(apply_type, axis=1)

            # df3.to_csv("entrust_all.csv", index=False)
            df7.to_csv(new_entrust_path, index=False, encoding="utf-8-sig")


        print("重建完成!") 
        return df

    except Exception as e:
        print("reconstruct_entrust_data() Exception: ", e)







# 将.parquet文件转换成.csv文件, 并执行上海逐笔委托重建
@measure_time
def convert_and_reconstruct(date, code):
    try:
        # 定义文件路径
        wt_file = f"data\\L2_Order\\{date}\\{code}.parquet"          # 逐笔委托原始数据
        cj_file = f"data\\L2Transaction\\{date}\\{code}.parquet"     # 逐笔成交原始数据
        # snap_file = f"data\\BIDASK\\{date}\\{code}.parquet"          # L1快照原始数据


        entrust_temp_file = f"data\\entrust_temp_{date}_{code}.csv"
        trade_file = f"data\\trade_{date}_{code}.csv"
        entrust_file = f"data\\entrust_{date}_{code}.csv"
      
        # 执行转换
        convert_parquet_files(wt_file, cj_file, entrust_temp_file, trade_file)


        # 执行重建
        df = reconstruct_entrust_data(entrust_temp_file, trade_file, entrust_file)
        # df.to_csv("merge.csv", index=False)   #将委托和成并合并流写入到CSV
        

    except Exception as e:
        print(f"convert_and_reconstruct() Exception: {e}")
   
  




# 使用示例
if __name__ == "__main__":


    date = "20260212"  
    code = "603090"  


    # 将.parquet文件转换成.csv文件, 并执行上海逐笔委托重建
    convert_and_reconstruct(date, code)



    # python order_reconstruct.py











