#!/usr/bin/python
# -*- coding:utf-8 -*-
#########################################################################
# File Name: upload.py
# Author: Bruce Zhang
# mail: zhangxb.sysu@gmail.com
# Created Time: 2015年11月26日 星期四 09时51分02秒
#########################################################################

import tornado.ioloop
import tornado.web
import tornado.httpserver
import os.path
import reco

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

filepath = ""

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", UploadFileHandler),
            (r"/reco", RecognizeHandler)
        ]
        settings = dict(
            template_path = os.path.join(os.path.dirname(__file__), "templates"),
            static_path = os.path.join(os.path.dirname(__file__), "static"),
            debug = True,
        )
        tornado.web.Application.__init__(self, handlers, **settings)

class UploadFileHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(
            "index.html",
            src = "static/images/preview.png",
            recoText = "nothing"
        )

    def post(self):
        global filepath
        if len(self.request.files) > 0:
            upload_path = os.path.join(os.path.dirname(__file__), 'static/files/')
            meta = self.request.files['file']
            meta = meta[0]
            filename = meta['filename']
            filepath = os.path.join(upload_path, filename)
            with open(filepath, 'wb') as up:
                up.write(meta['body'])
        else:
            filepath = "static/images/preview.png"
        self.render(
            "index.html",
            src = filepath,
            recoText = "nothing"
        )

class RecognizeHandler(tornado.web.RequestHandler):
    def get(self):
        global filepath
        print 'filepath = ', filepath
        if filepath == "":
            self.render(
                "index.html",
                src = "static/images/preview.png",
                recoText = "no image"
            )
        else:
            text = reco.doReco(filepath)
            self.render(
                "index.html",
                src = "static/images/rotatedRects.png",
                recoText = text
            )

if __name__ == '__main__':
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
