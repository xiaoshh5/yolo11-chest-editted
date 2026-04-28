# -*- coding: utf-8 -*-
# @Time    : 2023/7/11 18:29
# @Author  : 肆十二
# @Email   : 3048534499@qq.com
# @File    : mail
# @Software: PyCharm
from email.mime.text import MIMEText
from email.header import Header
import smtplib
from email.utils import *
def send_mail(user_mail="2392459641@qq.com" , info='当前位置发现火情，请及时查看'):
    my_sender = "1148392984@qq.com"  # 用户名
    my_pass = "phkxszqmuqakjhdg"  # 口令
    my_user = str(user_mail) # 收件人邮箱账号，我这边发送给自己
    ret = True
    try:
        msg = MIMEText(str(info), 'plain', 'utf-8')
        msg['From'] = formataddr(["Fire", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To'] = formataddr(["FK", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject'] = "火灾预警信息"  # 邮件的主题，也可以说是标题
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
        print("邮件发送成功")
    except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        print("邮件发送失败")
        ret = False
    return ret

if __name__ == '__main__':
    send_mail()
