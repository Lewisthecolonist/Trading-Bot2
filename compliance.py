class ComplianceChecker:
    def __init__(self, config):
        self.config = config

    def is_compliant(self, bid_price, ask_price, order_size):
        if not self.check_spread(bid_price, ask_price):
            return False
        if not self.check_order_size(order_size):
            return False
        if not self.check_daily_volume(order_size):
            return False
        return True

    def check_spread(self, bid_price, ask_price):
        spread = (ask_price - bid_price) / bid_price
        return spread <= self.config.MAX_SPREAD

    def check_order_size(self, order_size):
        return order_size <= self.config.MAX_ORDER_SIZE

    def check_daily_volume(self, order_size):
        # Implement logic to check if the order size doesn't exceed a certain percentage of daily volume
        # This would require fetching and storing daily volume data
        return True
