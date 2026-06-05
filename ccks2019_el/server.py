# encoding=utf8 
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import logging
import urlparse
import traceback
import el
import sys
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
reload(sys) 
sys.setdefaultencoding('utf8') 

class S(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With") 
        self.send_header('Content-type', 'text/json; charset=utf-8')
        self.end_headers()

    def _response(self, args):
        rtv = {'status':0, 'data':''}
        
        try:
            if args:
                args=urlparse.parse_qs(args).items()
                args=dict([(k,v[0]) for k,v in args])
            else:
                args={}

            # text=args.get("text","")

            el_data = []
            el_data.append(args)
            res = el.test(el_data)
            rtv["data"] = res

        except Exception as e:
            rtv["status"]=1
            rtv["msg"]='服务器错误：'+str(e)+"\n"+traceback.format_exc()
            
        try:
            rtv=json.dumps(rtv,ensure_ascii=False)
        except Exception as e:
            rtv={'status':2,'msg':'服务器返回数据错误：'+str(e)+"\n"+traceback.format_exc(),'data':''}
            rtv=json.dumps(rtv,ensure_ascii=False)
        
        self.do_HEAD()
        self.wfile.write(rtv.encode())
    
    def do_POST(self):
        args = self.rfile.read(int(self.headers['content-length']))

        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), args)

        self._response(args)


def run(server_class=HTTPServer, handler_class=S, port=3000):
    print("run()")
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("httpd.server_close()")
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    run()
