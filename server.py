import asyncio as aio 
from asyncio import Lock, Semaphore, Condition, Event

import zmq 
import zmq.asyncio as aiozmq  

import pickle 
import time 
import uvicorn

import multiprocessing as mp 
from os import path 

from fastapi import FastAPI, APIRouter, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.utils import get_openapi

from loguru import logger 

from api_schema import ServerStatus
from api_schema import EntrypointResponseModel, EntrypointResponseContentModel 
from api_schema import ComputeEmbeddingRequestModel, ComputeEmbeddingResponseModel, ComputeEmbeddingResponseContentModel

class APIServer:
    app_description = {
        "title":"text-semantic-embedding", 
        "version": 1.0, 
        "description":"""
            semantic embedding based on textrank and french_semantic
        """
    }

    def __init__(self, port:int, host:str, router_address:str, publisher_address:str, mounting_path:str, workers_barrier:mp.Barrier):
        
        self.port = port 
        self.host = host 
        self.mounting_path = mounting_path
        self.workers_barrier = workers_barrier

        self.router_address = router_address
        self.publisher_address = publisher_address        
        self.opened_sockets = 0 

        self.core = FastAPI(**self.app_description, docs_url='/')

        self.core.add_event_handler('startup', self.handle_startup)
        self.core.add_event_handler('shutdown', self.handle_shutdown)
        
        self.core.add_api_route(
            path='/heartbit', 
            endpoint=self.handle_entrypoint, 
            methods=['GET'], 
            response_model=EntrypointResponseModel
        )
        self.core.add_api_route(
            path='/compute_embedding', 
            endpoint=self.handle_compute_embedding, 
            methods=['POST'], 
            response_model=ComputeEmbeddingResponseModel
        )
    
    async def handle_startup(self):
        self.ctx = aiozmq.Context()
        self.ctx.set(zmq.MAX_SOCKETS, 32768)
        self.access_card = Semaphore(value=4096, loop=aio.get_event_loop())
        self.socket_mutex = Lock(loop=aio.get_event_loop())
        logger.debug('server is waiting for vectorizer to be ready')
        self.workers_barrier.wait()  # wait vectorizer to be ready...! 
        self.vectorizer_is_alive = Event(loop=aio.get_event_loop())
        self.vectorizer_is_alive.set()
        logger.success('server is up... zeromq was initialized')
    
    async def handle_shutdown(self):
        self.ctx.term()
        logger.success('server is down and has released its ressources')
        
    async def handle_entrypoint(self):
        return JSONResponse(
            status_code=200,
            content=EntrypointResponseModel(
                status=True,
                content=EntrypointResponseContentModel(
                    status=ServerStatus.ALIVE,
                    message='server is alive and ready to process incoming request'
                )
            ).dict()
        )
    
    async def handle_compute_embedding(self, incoming_req:ComputeEmbeddingRequestModel):
        async with self.access_card:
            async with self.socket_mutex:
                if not self.vectorizer_is_alive.is_set():
                    logger.warning(f'server can not process the incoming req : vectorizer is down')
                    return JSONResponse(
                        status_code=500,
                        content=ComputeEmbeddingResponseModel(
                            status=False,
                            request_id=incoming_req.request_id,
                            content=ComputeEmbeddingResponseContentModel(),
                            error_message='vectorizer is not alive ...!' 
                        ).dict()
                    )
                
            try:
                dealer_socket:aiozmq.Socket = self.ctx.socket(zmq.DEALER)
                dealer_socket.connect(self.router_address)

                subscriber_socket:aiozmq.Socket = self.ctx.socket(zmq.SUB)
                subscriber_socket.connect(self.publisher_address)
                subscriber_socket.setsockopt(zmq.SUBSCRIBE, b'EXIT_LOOP')

                poller = aiozmq.Poller()
                poller.register(dealer_socket, zmq.POLLIN)
                poller.register(subscriber_socket, zmq.POLLIN)
            except Exception as e:
                error = f'Error => {e}'
                logger.error(error)
                
                return JSONResponse(
                    status_code=500,
                    content=ComputeEmbeddingResponseModel(
                        status=False,
                        request_id=incoming_req.request_id,
                        content=ComputeEmbeddingResponseContentModel(),
                        error_message=error 
                    ).dict()
                )
            
            async with self.socket_mutex:
                self.opened_sockets += 1
                logger.debug(f'server get a new request for making embedding...! {self.opened_sockets}')

            map_loop_status2error = {
                0: 'TIMEOUT... this request take too long time to be processed >= 60s',
                2: 'Vectorizer is not available',
                3: 'Vectorizer was not able to compute the embedding' 
            }

            try:
                await dealer_socket.send(b'', flags=zmq.SNDMORE)
                await dealer_socket.send_string(incoming_req.text)
                
                counter = 0
                keep_loop = 0 
                while keep_loop == 0 and counter < 1200000:
                    map_socket2status = dict(await poller.poll(100))
                    dealer_polled_status = map_socket2status.get(dealer_socket, None)
                    subscriber_polled_status = map_socket2status.get(subscriber_socket, None)
                    if dealer_polled_status == zmq.POLLIN:
                        _, encoded_rsp = await dealer_socket.recv_multipart()
                        embedding = pickle.loads(encoded_rsp)
                        logger.debug(f'server has finished to process the embedding request...!')
                        if embedding is None:
                            keep_loop = 3 
                        else:
                            keep_loop = 1  # successfull response 
                    if subscriber_polled_status == zmq.POLLIN:
                        topic, _ = await subscriber_socket.recv_multipart()
                        if topic == b'EXIT_LOOP':
                            async with self.socket_mutex:
                                if self.vectorizer_is_alive.is_set():
                                    self.vectorizer_is_alive.clear()
                            keep_loop = 2 
                    counter += 100 # 100ms 
                # end while loop 

                poller.unregister(dealer_socket)
                poller.unregister(subscriber_socket)
                dealer_socket.close(linger=0)
                subscriber_socket.close(linger=0)

                async with self.socket_mutex:
                    self.opened_sockets -= 1
                    
                if keep_loop in [0, 2, 3]:
                    return JSONResponse(
                        status_code=500,
                        content=ComputeEmbeddingResponseModel(
                            status=False,
                            request_id=incoming_req.request_id,
                            content=None,
                            error_message=map_loop_status2error[keep_loop]
                        ).dict()
                    )
                
                assert keep_loop == 1 
                return JSONResponse(
                    status_code=200,
                    content=ComputeEmbeddingResponseModel(
                        status=True,
                        request_id=incoming_req.request_id,
                        content=ComputeEmbeddingResponseContentModel(
                            embedding=embedding.tolist()
                        )
                    ).dict()
                )

            except Exception as e:
                error = f'Error => {e}'
                logger.error(error)

                poller.unregister(dealer_socket)
                poller.unregister(subscriber_socket)
                dealer_socket.close(linger=0)
                subscriber_socket.close(linger=0)
                
                async with self.socket_mutex:
                    self.opened_sockets -= 1
                
                return JSONResponse(
                    status_code=500,
                    content=ComputeEmbeddingResponseModel(
                        status=False,
                        request_id=incoming_req.request_id,
                        content=ComputeEmbeddingResponseContentModel(),
                        error_message=error 
                    ).dict()
                )
        # end semaphore acess card context manager 

    def start(self):
        uvicorn.run(app=self.core, port=self.port, host=self.host, root_path=self.mounting_path)

def start_server(port:int, hostname:str, router_address:str, publisher_address:str, mounting_path:str, workers_barrier:mp.Barrier):
    server_ = APIServer(port=port, host=hostname, router_address=router_address, publisher_address=publisher_address, mounting_path=mounting_path, workers_barrier=workers_barrier)
    server_.start()