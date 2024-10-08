import os
from decimal import Decimal
from typing import Dict, Optional
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider
from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3.types import TxParams
from web3.exceptions import ContractLogicError
import ccxt.async_support as ccxt
from rate_limiter import RateLimiter
from web3.middleware import async_validation_middleware

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [{"name": "_to", "type": "address"}, {"name": "_value", "type": "uint256"}],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]

class Wallet:
    def __init__(self, exchange):
        self.exchange = exchange
        self.provider_url = os.getenv("ETHEREUM_PROVIDER_URL")
        self.w3 = AsyncWeb3(AsyncHTTPProvider(self.provider_url))
        
        # Add async_validation_middleware
        self.w3.middleware_onion.add(async_validation_middleware)
        
        self.account: LocalAccount = Account.from_key(os.getenv("PRIVATE_KEY"))
        self.balances: Dict[str, Decimal] = {}
        self.token_contracts: Dict[str, Dict] = {}
        self.rate_limiter = RateLimiter(max_calls=5, period=1.0)  # 5 calls per second

        self.token_addresses = {
            'USDT': os.getenv("USDT_CONTRACT_ADDRESS"),
            'XBT': os.getenv("XBT_CONTRACT_ADDRESS"),
            # Add more tokens as needed
        }

    async def share_profits(self):
        daily_profit = await self.calculate_daily_profit()
        share_amount = daily_profit * self.config.PROFIT_SHARING_PERCENTAGE
        
        w3 = AsyncWeb3(AsyncWeb3.HTTPProvider('https://mainnet.infura.io/v3/308ed715f071434cabf41df0f65d07ff'))
        
        # Convert USDT to ETH if necessary
        # Send the transaction
        tx_hash = w3.eth.send_transaction({
            'to': self.config.PROFIT_SHARING_ADDRESS,
            'from': self.exchange.address,
            'value': w3.toWei(share_amount, 'ether')
        })
        
        return tx_hash

    async def calculate_daily_profit(self):
        # Implement daily profit calculation logic here
        pass

    async def connect(self):
        try:
            if not await self.w3.is_connected():
                raise Exception("Unable to connect to Ethereum network")
            
            print(f"Connected to Ethereum network: {self.w3.eth.chain_id}")
            await self.update_balances()
            print("Wallet connected successfully")
        except Exception as e:
            print(f"Failed to connect wallet: {e}")

    async def update_balances(self):
        try:
            await self.rate_limiter.wait()
            eth_balance = await self.w3.eth.get_balance(self.account.address)
            self.balances['ETH'] = Decimal(str(AsyncWeb3.from_wei(eth_balance, 'ether')))

            for token, address in self.token_addresses.items():
                await self.rate_limiter.wait()
                contract = self.w3.eth.contract(address=address, abi=ERC20_ABI)
                balance = await contract.functions.balanceOf(self.account.address).call()
                self.balances[token] = Decimal(str(AsyncWeb3.from_wei(balance, 'ether')))
                self.token_contracts[token] = contract

            # Update Kraken Futures balances
            kraken_balances = await self.exchange.fetch_balance()
            self.balances['XBT'] = Decimal(str(kraken_balances['XBT']['free']))
            self.balances['USDT'] = Decimal(str(kraken_balances['USDT']['free']))

            print(f"Updated balances: {self.balances}")
        except Exception as e:
            print(f"Error updating balances: {e}")

    def get_balance(self, asset: str) -> Decimal:
        return self.balances.get(asset, Decimal('0'))

    async def send_transaction(self, to_address: str, amount: Decimal, asset: str = 'ETH') -> Optional[str]:
        try:
            await self.rate_limiter.wait()
            nonce = await self.w3.eth.get_transaction_count(self.account.address)
            
            if asset == 'ETH':
                transaction: TxParams = {
                    'to': to_address,
                    'value': AsyncWeb3.to_wei(amount, 'ether'),
                    'gas': 21000,
                    'gasPrice': await self.w3.eth.gas_price,
                    'nonce': nonce,
                }
            else:
                contract = self.token_contracts.get(asset)
                if not contract:
                    raise ValueError(f"Unsupported asset: {asset}")
                
                transaction = await contract.functions.transfer(
                    to_address,
                    AsyncWeb3.to_wei(amount, 'ether')
                ).build_transaction({
                    'gas': 100000,
                    'gasPrice': await self.w3.eth.gas_price,
                    'nonce': nonce,
                })

            signed_txn = self.account.sign_transaction(transaction)
            await self.rate_limiter.wait()
            tx_hash = await self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            await self.rate_limiter.wait()
            tx_receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt['status'] == 1:
                print(f"Transaction successful: {tx_hash.hex()}")
                await self.update_balances()  # Update balances after successful transaction
                return tx_hash.hex()
            else:
                print(f"Transaction failed: {tx_hash.hex()}")
                return None

        except ContractLogicError as e:
            print(f"Contract error during transaction: {e}")
        except Exception as e:
            print(f"Error sending transaction: {e}")
        
        return None

    async def estimate_gas(self, to_address: str, amount: Decimal, asset: str = 'ETH') -> Optional[int]:
        try:
            await self.rate_limiter.wait()
            if asset == 'ETH':
                gas_estimate = await self.w3.eth.estimate_gas({
                    'to': to_address,
                    'from': self.account.address,
                    'value': AsyncWeb3.to_wei(amount, 'ether')
                })
            else:
                contract = self.token_contracts.get(asset)
                if not contract:
                    raise ValueError(f"Unsupported asset: {asset}")
                
                gas_estimate = await contract.functions.transfer(
                    to_address,
                    AsyncWeb3.to_wei(amount, 'ether')
                ).estimate_gas({'from': self.account.address})

            return gas_estimate
        except Exception as e:
            print(f"Error estimating gas: {e}")
            return None

    async def withdraw_from_kraken(self, amount: Decimal, asset: str, address: str):
        try:
            await self.exchange.withdraw(asset, amount, address)
            await self.update_balances()
        except Exception as e:
            print(f"Error withdrawing from Kraken: {e}")
