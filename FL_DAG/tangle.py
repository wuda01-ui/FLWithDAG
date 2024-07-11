import json
import os
import random
from multiprocessing import Pool, Process, current_process

import numpy as np

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

from .transaction import Transaction
from .node import Node
from .poison_type import PoisonType

class Tangle:
    def __init__(self, transactions, genesis):
        self.transactions = transactions
        self.genesis = genesis
        if current_process().name == 'MainProcess':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            self.process_pool = Pool(10)

    def add_transaction(self, tip):
        self.transactions[tip.name()] = tip

    def run_nodes(self, train_fn, clients, rnd, num_epochs=1, batch_size=10, malicious_clients=None, poison_type=PoisonType.NONE):
        norm_this_round = []
        new_transactions = []

        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}

        train_params = [[client.id, client.group, client.model.flops, random.randint(0, 4294967295), client.train_data, client.eval_data, self.name, (client.id in malicious_clients), poison_type] for client in clients]

        results = self.process_pool.starmap(train_fn, train_params)

        for tx, metrics, client_id, client_sys_metrics in results:
            if tx is None:
                continue

            sys_metrics[client_id][BYTES_READ_KEY] += client_sys_metrics[BYTES_READ_KEY]
            sys_metrics[client_id][BYTES_WRITTEN_KEY] += client_sys_metrics[BYTES_WRITTEN_KEY]
            sys_metrics[client_id][LOCAL_COMPUTATIONS_KEY] = client_sys_metrics[LOCAL_COMPUTATIONS_KEY]

            tx.tag = rnd
            new_transactions.append(tx)

        for tx in new_transactions:
            self.add_transaction(tx)

        return sys_metrics

    def test_model(self, test_fn, clients_to_test, set_to_use='test'):
        metrics = {}

        test_params = [[client.id, client.group, client.model.flops, random.randint(0, 4294967295), client.train_data, client.eval_data, self.name, set_to_use] for client in clients_to_test]
        results = self.process_pool.starmap(test_fn, test_params)

        for client, c_metrics in results:
            metrics[client] = c_metrics

        return metrics

    def save(self, tangle_name, global_loss, global_accuracy, norm):
        n = [{'name': t.name(), 'time': t.tag, 'malicious': t.malicious, 'parents': list(t.parents)} for _, t in self.transactions.items()]

        with open(f'tangle_data/tangle_{tangle_name}.json', 'w') as outfile:
            json.dump({'nodes': n, 'genesis': self.genesis, 'global_loss': global_loss, 'global_accuracy': global_accuracy, 'norm': norm}, outfile)

        self.name = tangle_name

    @classmethod
    def fromfile(cls, tangle_name):
      with open(f'tangle_data/tangle_{tangle_name}.json', 'r') as tanglefile:
          t = json.load(tanglefile)

      transactions = {n['name']: Transaction(None, set(n['parents']), n['name'], n['time'], n['malicious'] if 'malicious' in n else False) for n in t['nodes']}
      tangle = cls(transactions, t['genesis'])
      tangle.name = tangle_name
      return tangle
