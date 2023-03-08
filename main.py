import click 
import uvicorn

from loguru import logger 
from typing import List, Tuple, Dict, Optional 

from os import path 
from glob import glob
from time import perf_counter, sleep 

import multiprocessing as mp 

from server import start_server
from vectorizer import start_vectorizer

@click.group(chain=False, invoke_without_command=True)
@click.option('--transformers_cache', envvar='TRANSFORMERS_CACHE', required=True)
@click.pass_context
def command_line_interface(ctx:click.core.Context, transformers_cache:str):
    ctx.ensure_object(dict)
    ctx.obj['cache_folder'] = transformers_cache
    crr_command = ctx.invoked_subcommand
    if crr_command is not None:
        logger.debug(f'{crr_command} was called')

@command_line_interface.command()
@click.option('--port', type=int)
@click.option('--hostname', type=str)
@click.option('--transformer_model_name', type=str, default='Sahajtomar/french_semantic')
@click.option('--language_model_name', type=str, default='fr_core_news_md')
@click.option('--router_address', type=str, default='ipc://router.ipc')
@click.option('--publisher_address', type=str, default='ipc://publisher.ipc')
@click.option('--mounting_path', type=str, default='/')
@click.pass_context
def start_services(ctx:click.core.Context, port:int, hostname:str, transformer_model_name:str, language_model_name:str, router_address:str, publisher_address:str, mounting_path:str):
    assert mounting_path.startswith('/')
    cache_folder = ctx.obj['cache_folder']
    
    workers_barrier = mp.Barrier(parties=2)

    server_process = mp.Process(
        target=start_server, 
        kwargs={
            'port': port, 
            'hostname': hostname, 
            'mounting_path': mounting_path, 
            'router_address': router_address, 
            'publisher_address': publisher_address,
            'workers_barrier': workers_barrier
        }
    ) 
    
    vectorizer_process = mp.Process(
        target=start_vectorizer,
        kwargs={
            'transformer_model_name': transformer_model_name,
            'language_model_name': language_model_name, 
            'cache_folder': cache_folder,
            'router_address': router_address, 
            'publisher_address': publisher_address,
            'workers_barrier': workers_barrier
        }
    )

    try:
        vectorizer_process.start()
        server_process.start()

        keep_loop = True 
        while keep_loop:
            sleep(1)
            if not vectorizer_process.is_alive() or not server_process.is_alive():
                keep_loop = False 
        # ...!
        logger.warning('some problems to start all workrs')
        workers_barrier.abort()
    except KeyboardInterrupt:
        logger.debug('ctl+c ...!') 
        vectorizer_process.join()
        server_process.join()
    except Exception as e:
        logger.error(e)
    finally:
        vectorizer_process.terminate()
        server_process.terminate()
        vectorizer_process.join()
        server_process.join()

if __name__ == '__main__':
    command_line_interface()