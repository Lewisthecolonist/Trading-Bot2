from enum import Enum, auto

class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()

class Event:
    pass

class MarketEvent(Event):
    def __init__(self, timestamp, data):
        self.type = EventType.MARKET
        self.timestamp = timestamp
        self.data = data

class SignalEvent(Event):
    def __init__(self, timestamp, symbol, signal):
        self.type = EventType.SIGNAL
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal = signal

class OrderEvent(Event):
    def __init__(self, timestamp, symbol, order_type, quantity, direction):
        self.type = EventType.ORDER
        self.timestamp = timestamp
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

class FillEvent(Event):
    def __init__(self, timestamp, symbol, exchange, quantity, direction, fill_cost, commission=None):
        self.type = EventType.FILL
        self.timestamp = timestamp
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission
