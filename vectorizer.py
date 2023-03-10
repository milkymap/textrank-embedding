import zmq 

import numpy as np 
import multiprocessing as mp 

import torch as th 
import operator as op 

from time import sleep, perf_counter
from libraries.log import logger 
from libraries.strategies import (
    load_language_model, load_sentence_transformer, 
    compute_pairwise_matrix,
    to_sentences
)

#docker run --rm -it --name embedding -v /home/ibrahima/Volume/transformers_cache/:/home/solver/transformers_cache -p 8090:8000 embedding:gpu-0.0 start-services --port 8000 --hostname '0.0.0.0' --mounting_path '/'


class ZMQVectorizer:
    def __init__(self, workers_barrier:mp.Barrier, router_address:str, publisher_address:str, transformer_model_name:str, language_model_name:str, cache_folder:str, gpu_index:int):
        self.workers_barrier = workers_barrier
        self.gpu_index = gpu_index
        self.router_address = router_address
        self.publisher_address = publisher_address

        self.language_model_name = language_model_name
        self.transformer_model_name = transformer_model_name 
        self.cache_folder = cache_folder
        self.zeromq_initialized = 0 

    def start_loop(self):
        if self.zeromq_initialized == 0:
            return 0 

        logger.debug('vectorizer starts the event loop')        
        keep_loop = True 
        while keep_loop:
            try:
                polled_status = self.router_socket.poll(100)  # 100ms 
                if polled_status == zmq.POLLIN:
                    client_address, _, encoded_text = self.router_socket.recv_multipart()
                    try:
                        text = encoded_text.decode('utf-8')
                        sentences = to_sentences(text, self.language_model, valid_length=1)    
                        if len(sentences) == 0:  # can not split text to sentences 
                            embedding = self.transformer_model.encode(text, device=self.device)
                        else: 
                            embeddings = self.transformer_model.encode(sentences, device=self.device)
                            if len(sentences) == 1:
                                embedding = embeddings[0]
                            else:
                                nb_nodes = len(sentences)
                                adjacency_matrix = compute_pairwise_matrix(embeddings)
                                scores = np.sum(adjacency_matrix, axis=1) / nb_nodes 
                                embedding = np.mean(embeddings * scores[:, None], axis=0)
                    except Exception as e:
                        logger.error(e)
                        embedding = None 
                    self.router_socket.send_multipart([client_address, b''], flags=zmq.SNDMORE)
                    self.router_socket.send_pyobj(embedding)
            except KeyboardInterrupt:
                keep_loop = False 
                logger.warning('vectorizer get the ctl+c signal')
            except Exception as e:
                logger.error(e)
        # end while loop ...!

        logger.debug('vectorizer quits the event loop')
        self.publisher_socket.send(b'EXIT_LOOP', flags=zmq.SNDMORE)
        self.publisher_socket.send(b'')
        sleep(0.1)  # wait 10ms 

    def __enter__(self):
        try:
            logger.debug('vectorizer will load the language model')
            self.language_model = load_language_model(self.language_model_name)
            logger.debug('vectotorize is loading the transformers model')
            logger.debug(f'number of detected gpu card : {th.cuda.device_count()}')
            self.device = th.device('cpu' if not th.cuda.is_available() else f'cuda:{self.gpu_index}')
            self.transformer_model = load_sentence_transformer(self.transformer_model_name, self.cache_folder, device=self.device)
            logger.success('vectorizer has finished to load the transformers and the language model')
            logger.success(f'transformer is on the device : {self.device}')
        except Exception as e:
            logger.error(e)
        else:
            try:
                self.ctx = zmq.Context()
                self.router_socket:zmq.Socket = self.ctx.socket(zmq.ROUTER)
                self.router_socket.bind(self.router_address)

                self.publisher_socket:zmq.Socket = self.ctx.socket(zmq.PUB)
                self.publisher_socket.bind(self.publisher_address)

                self.zeromq_initialized = 1 
                logger.success('vectorizer has initialized its zeromq ressources')
                self.workers_barrier.wait()  # wait the server ...! 
            except Exception as e:
                logger.error(e)

        return self 
    
    def __exit__(self, exc, val, traceback):
        if self.zeromq_initialized == 1:
            self.publisher_socket.close(linger=0)
            self.router_socket.close(linger=0)
            self.ctx.term()
            logger.success('vectorizer has reeased all zeromq ressources')
        logger.debug('vectorizer shutdown...!')

def start_vectorizer(transformer_model_name:str, cache_folder:str, language_model_name:str, router_address:str, publisher_address:str, workers_barrier:mp.Barrier, gpu_index:int):
    with ZMQVectorizer(router_address=router_address, publisher_address=publisher_address, transformer_model_name=transformer_model_name, language_model_name=language_model_name, cache_folder=cache_folder, workers_barrier=workers_barrier, gpu_index=gpu_index) as agent:
        agent.start_loop()
