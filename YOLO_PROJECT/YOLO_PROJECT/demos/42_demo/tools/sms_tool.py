# -*-coding:utf-8 -*-

"""
#-------------------------------
# @Author : 肆十二
# @QQ : 3045834499 可定制毕设
#-------------------------------
# @File : duanx.py
# @Description: 文件描述
# @Software : PyCharm
# @Time : 2024/4/3 1:03
#-------------------------------
"""
# coding=utf-8
import urllib
import urllib.request
import hashlib

def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

def send_phone():
    statusStr = {
        '0': '短信发送成功',
        '-1': '参数不全',
        '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
        '30': '密码错误',
        '40': '账号不存在',
        '41': '余额不足',
        '42': '账户已过期',
        '43': 'IP地址限制',
        '50': '内容含有敏感词'
    }

    smsapi = "http://api.smsbao.com/"
    # 短信平台账号
    user = 'songcm'
    # 短信平台密码
    password = md5('d2c6da583d1248d99402a5dbae2898f7')
    # 要发送的短信内容
    content = '发现火情，请及时处理！'
    # 要发送短信的手机号码
    phone = '18085836954'

    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print(statusStr[the_page])

if __name__ == '__main__':
    send_phone()