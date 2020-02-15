#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: fetch_captcha.py
# Created Date: Wednesday, January 8th 2020, 12:05:13 pm
# Author: Rabbit
# -------------------------
# Copyright (c) 2020 Rabbit
# --------------------------------------------------------------------
###

import os
import time
from requests.sessions import Session
from requests.exceptions import RequestException

from const import DOWNLOAD_DIR
from utils import md5


TIMEOUT = 10
INTERVAL = 1.5
ERROR_INTERVAL = 10


def main():
    
    s = Session()
    s.headers = {
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Host": "iaaa.pku.edu.cn",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/78.0.3904.97 Chrome/78.0.3904.97 Safari/537.36",
        "Referer": "https://iaaa.pku.edu.cn/iaaa/oauth.jsp?appID=portal2017&appName=%E5%8C%97%E4%BA%AC%E5%A4%A7%E5%AD%A6%E6%A0%A1%E5%86%85%E4%BF%A1%E6%81%AF%E9%97%A8%E6%88%B7%E6%96%B0%E7%89%88&redirectUrl=https://portal.pku.edu.cn/portal2017/ssoLogin.do",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-origin",
    }

    while True:
        
        try:
            r = s.get("https://iaaa.pku.edu.cn/iaaa/servlet/DrawServlet?Rand=", timeout=TIMEOUT)
        except RequestException as e:
            print(e)
            time.sleep(ERROR_INTERVAL)
            continue

        filename = "%s.jpg" % md5(r.content)
        path = os.path.normpath(os.path.join(DOWNLOAD_DIR, filename))

        with open(path, "wb") as fp:
            fp.write(r.content)

        print("save captcha to %s" % path)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
    