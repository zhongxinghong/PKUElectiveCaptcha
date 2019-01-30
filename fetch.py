#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: fetch.py

import os
import time
import requests
from requests.exceptions import HTTPError, ProxyError, ConnectTimeout,\
        ReadTimeout, ConnectionError, ChunkedEncodingError
from util import Download_Dir, Raw_Captcha_Dir, MD5

Proxy_Errors = (ProxyError, ConnectTimeout, ReadTimeout, ConnectionError, ChunkedEncodingError)


class ProxyAPIClient(object):

    Host     = "127.0.0.1"
    Port     = "7070"
    Base_URL = "http://{}:{}".format(Host,Port)

    Root_URL            = Base_URL + "/"
    Count_Proxy_URL     = Base_URL + "/count"
    Get_Proxy_URL       = Base_URL + "/get"
    Get_Pool_Status_URL = Base_URL + "/status"

    def __init__(self):
        self._session = requests.session()

    def root(self):
        return self._session.get(self.Root_URL).text

    def get(self):
        return self._session.get(self.Get_Proxy_URL).text

    def count(self):
        return self._session.get(self.Count_Proxy_URL).text

    def status(self):
        return self,_session.get(self.Get_Pool_Status_URL).text


class CaptchaFetchClient(object):

    Default_Timeout = 5

    def __init__(self):
        self._session = requests.session()
        self._session.headers.update({
                "Host": "elective.pku.edu.cn",
                "Upgrade-Insecure-Requests": "1",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0",
            })

    def _get(self, *args, timeout=Default_Timeout, **kwargs):
        return self._session.get(*args, timeout=timeout, **kwargs)

    def get_captcha(self, proxies={}):
        resp = self._get("http://elective.pku.edu.cn/elective2008/DrawServlet", proxies=proxies)
        self.save(resp.content)

    @staticmethod
    def save(imgBytes):
        file = MD5(imgBytes) + ".jpg"
        path = os.path.join(Download_Dir, file)
        with open(path, "wb") as fp:
            fp.write(imgBytes)
        print("get captcha " + file)


proxyAPIClient = ProxyAPIClient()
captchaFetchClient = CaptchaFetchClient()

proxy = proxyAPIClient.get()

while True:
    print("Use Proxy " + proxy)
    try:
        img = captchaFetchClient.get_captcha(proxies=dict([proxy.split("://")]))
    except Proxy_Errors as e:
        proxy = proxyAPIClient.get()
    except HTTPError as e:
        raise e
    except Exception as e:
        raise e
