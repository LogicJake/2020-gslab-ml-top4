# 2020-gslab-ml-top4
2020腾讯游戏安全技术竞赛机器学习组优秀奖源码
===============================================================================================================
**2020腾讯游戏安全技术竞赛机器学习组优秀奖（LightGBM单模）**

## 主办方：腾讯游戏安全实验室  
## 赛道：2020-腾讯游戏安全技术竞赛机器学习组  
**赛道链接**：https://gslab.qq.com/html/competition/2020/index.htm  
**赛程时间**：2019.04.03-2020.04.10  
**百度云盘下载链接**：为避免数据丢失，提供数据集下载地址链接：https://pan.baidu.com/s/1yhXwmtyap-CR2aj8qq_Pag  密码: mg41

## 1.数据说明  
* 数据来自某MMORPG(大型多人在线角色扮演游戏), 并经过脱敏处理
* 日期
  * 2020年03月01日的数据(含标签) 为训练集
  * 2020年03月05日的数据为测试集
* 基础知识
 *uin: 唯一标识游戏内的一个用户, 比如你的qq或微信
  * roleid: 一个uin可能有多个角色
* 存储格式
  * 文件名为: 年月日.txt
  * 以文本存储
  * 以竖线|分隔
  * 空 或 \N 表示数据缺失
* 目录说明
  * label_black 黑标签: 打金工作室帐号
  * label_white 白标签: 非打金工作室帐号
  * role_login 角色登入游戏
  * role_logout 角色登出游戏
  * role_create 创建新角色
  * uin_chat 按天统计的帐号发言次数
  * 以下数据仅在决赛时提供
    * role_moneyflow 角色的详细金钱流水信息(当天按时间顺序前300条记录)
    * role_itemflow 角色的详细物品流水信息(当天按时间顺序前300条记录)

### role_login 角色登入游戏
|  # |     列名    |  类型  |             备注             |
|:--:|:-----------:|:------:|:----------------------------:|
|  1 | dteventtime | STRING | 时间,格式YYYY-MM-DD HH:MM:SS |
|  2 |    platid   | BIGINT |        ios=0/android=1       |
|  3 |    areaid   | BIGINT |      微信=1/手Q=2/游客=3     |
|  4 |   worldid   | BIGINT |       游戏小区(已加密)       |
|  5 |     uin     | STRING |        openid(已加密)        |
|  6 |    roleid   | STRING |        角色id(已加密)        |
|  7 |   rolename  | STRING |        角色名(已置空)        |
|  8 |     job     | STRING |             职业             |
|  9 |  rolelevel  | BIGINT |             等级             |
| 10 |    power    | BIGINT |             战力             |
| 11 |  friendsnum | BIGINT |           好友数量           |
| 12 |   network   | STRING |        3G/WIFI/2G/NULL       |
| 13 |   clientip  | STRING |       客户端IP(已加密)       |
| 14 |   deviceid  | STRING |        设备ID(已加密)        |

### role_logout 角色登出游戏
|  # |     列名    |  类型  |             备注             |
|:--:|:-----------:|:------:|:----------------------------:|
|  1 | dteventtime | STRING | 时间,格式YYYY-MM-DD HH:MM:SS |
|  2 |    platid   | BIGINT |        ios=0/android=1       |
|  3 |    areaid   | BIGINT |      微信=1/手Q=2/游客=3     |
|  4 |   worldid   | BIGINT |       游戏小区(已加密)       |
|  5 |     uin     | STRING |        openid(已加密)        |
|  6 |    roleid   | STRING |        角色id(已加密)        |
|  7 |   rolename  | STRING |        角色名(已置空)        |
|  8 |     job     | STRING |             职业             |
|  9 |  rolelevel  | BIGINT |             等级             |
| 10 |    power    | BIGINT |             战力             |
| 11 |  friendsnum | BIGINT |           好友数量           |
| 12 |   network   | STRING |        3G/WIFI/2G/NULL       |
| 13 |   clientip  | STRING |       客户端IP(已加密)       |
| 14 |   deviceid  | STRING |        设备ID(已加密)        |
| 15 |  onlinetime | BIGINT |         在线时长(秒)         |

### role_create 创建新角色
|  # |     列名    |  类型  |         备注        |
|:--:|:-----------:|:------:|:-------------------:|
|  1 | dteventtime | STRING | YYYY-MM-DD HH#MM#SS |
|  2 |    platid   | BIGINT |   ios=0/android=1   |
|  3 |    areaid   | BIGINT | 微信=1/手Q=2/游客=3 |
|  4 |   worldid   | BIGINT |   游戏小区(已加密)  |
|  5 |     uin     | STRING |    openid(已加密)   |
|  6 |    roleid   | STRING |    角色id(已加密)   |
|  7 |   rolename  | STRING |    角色名(已置空)   |
|  8 |     job     | STRING |         职业        |
|  9 |  regchannel | STRING |       注册渠道      |
| 10 |   network   | STRING |      3G/WIFI/2G     |
| 11 |   clientip  | STRING |   客户端IP(已加密)  |
| 12 |   deviceid  | STRING |    设备ID(已加密)   |

### uin_chat 按天统计的帐号发言次数
| # |   列名   |  类型  |      备注      |
|:-:|:--------:|:------:|:--------------:|
| 1 |    uin   | STRING | openid(已加密) |
| 2 | chat_cnt | BIGINT |    发言条数    |

### role_moneyflow 帐号的详细金钱流水信息
|  # |     列名    |  类型  |             备注             |
|:--:|:-----------:|:------:|:----------------------------:|
|  1 | dteventtime | STRING | 时间,格式YYYY-MM-DD HH:MM:SS |
|  2 |   worldid   | BIGINT |       游戏小区(已加密)       |
|  3 |     uin     | STRING |        openid(已加密)        |
|  4 |    roleid   | STRING |        角色id(已加密)        |
|  5 |  rolelevel  | BIGINT |             等级             |
|  6 |  iMoneyType | STRING |           货币类型           |
|  7 |    iMoney   | BIGINT |          货币变化数          |
|  8 |  AfterMoney | BIGINT |       动作后的货币存量       |
|  9 | AddOrReduce | BIGINT |       货币增加 0/减少 1      |
| 10 |    Reason   | STRING |       货币流动一级原因       |
| 11 |  SubReason  | STRING |       货币流动二级原因       |

### role_itemflow 帐号的详细物品流水信息
|  # |     列名    |  类型  |             备注             |
|:--:|:-----------:|:------:|:----------------------------:|
|  1 | dteventtime | STRING | 时间,格式YYYY-MM-DD HH:MM:SS |
|  2 |   worldid   | BIGINT |       游戏小区(已加密)       |
|  3 |     uin     | STRING |        openid(已加密)        |
|  4 |    roleid   | STRING |        角色id(已加密)        |
|  5 |  rolelevel  | BIGINT |             等级             |
|  6 |   Itemtype  | STRING |           道具类型           |
|  7 |    Itemid   | STRING |            道具ID            |
|  8 |    Count    | BIGINT |         道具变动数量         |
|  9 |  Aftercount | BIGINT |      动作后道具剩余数量      |
| 10 | Addorreduce | BIGINT |         增加 0/减少 1        |
| 11 |    Reason   | STRING |       道具流动一级原因       |
| 12 |  SubReason  | STRING |       道具流动二级原因       |

## 2.配置环境与依赖库 
  - python3
  - scikit-learn
  - gensim
  - Ubuntu   
  - LightGBM

## 3.运行代码说明  
python main.py --data_dir 数据所在文件夹路径 --res_file 结果保存文件

## ４.模型训练   
单模，线下0.957左右
