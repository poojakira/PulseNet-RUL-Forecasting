import hashlib
import json
import time
import os
import numpy as np

# Helper to handle non-serializable numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Block:
    def __init__(self, index, timestamp, data, previous_hash, hash=None):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        # If loading from file, use existing hash, otherwise calculate new
        self.hash = hash if hash else self.calculate_hash()

    def calculate_hash(self):
        # Sort keys ensures consistent hashing
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True, cls=NpEncoder)
        
        return hashlib.sha256(block_string.encode()).hexdigest()

class BlackBoxLedger:
    def __init__(self, chain_file="blackbox_ledger.json"):
        self.chain_file = chain_file
        self.chain = []
        
        if os.path.exists(chain_file):
            self.load_chain()
        else:
            self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, time.time(), "GENESIS_BLOCK_ENGINE_START", "0")
        self.chain.append(genesis_block)
        self.save_chain()

    def add_entry(self, unit_id, cycles, health_score, status):
        previous_block = self.chain[-1]
        
        # Payload: The critical data we want to prove wasn't tampered with
        data_payload = {
            "unit_id": unit_id,
            "cycles": cycles,
            "health_score": round(health_score, 2),
            "status": status
        }

        new_block = Block(
            index=previous_block.index + 1,
            timestamp=time.time(),
            data=data_payload,
            previous_hash=previous_block.hash
        )
        
        self.chain.append(new_block)
        self.save_chain()
        return new_block.hash

    def save_chain(self):
        chain_data = [b.__dict__ for b in self.chain]
        with open(self.chain_file, "w") as f:
            json.dump(chain_data, f, indent=4, cls=NpEncoder)

    def load_chain(self):
        with open(self.chain_file, "r") as f:
            chain_data = json.load(f)
            self.chain = [Block(**data) for data in chain_data]

    def validate_integrity(self):
        """
        Re-calculates hashes for the entire chain to detect tampering.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]

            # 1. Check if the stored hash matches the calculated hash of the data
            if current.hash != current.calculate_hash():
                return False, f"Block #{current.index} Data Tampered!"
            
            # 2. Check if the chain link is broken
            if current.previous_hash != previous.hash:
                return False, f"Broken Chain Link at Block #{current.index}"

        return True, "Ledger Secure"