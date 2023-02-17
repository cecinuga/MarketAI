import sys
sys.path.insert(0, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\lib\\')
sys.path.insert(1, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\Models\\')
sys.path.insert(2, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\Models\\components')
import Layer
import Network
from datetime import datetime
from run import *
import dxfeed as dx
import pandas as pd
from graphics import Graphics
from preprocessing import *
from termcolor import colored, cprint
from dxfeed.core import DXFeedPy as dxc
from dateutil.relativedelta import relativedelta
from dxfeed.core.utils.handler import DefaultHandler

DXFEED_URL = 'demo.dxfeed.com:7300'
INDEX_STANDARD = colored(Graphics.INDEX_STANDARD, 'green', attrs=["reverse"])

def main():
    connection = dxc.dxf_create_connection(DXFEED_URL)
    connection_print = colored("Connected", 'green')
    print(f"{INDEX_STANDARD} connection: {DXFEED_URL} is {connection_print}")

    #sub = connection.dxf_create_subscription(connection, 'Quote')
    sub = dxc.dxf_create_subscription_timed(connection, 'Quote', int((datetime.now() - relativedelta(days=3)).timestamp()))
    handler = DefaultHandler()
    sub.set_event_handler(handler)
    dxc.dxf_attach_listener(sub)
    dxc.dxf_add_symbols(sub, ['AAPL'])
    handler.get_list()[-5:]


if __name__ == '__main__':
    main()
