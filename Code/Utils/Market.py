import time
import copy
import numpy as np

class Market:
    def __init__(self, network_obj, limits_obj):
        np.random.seed(14)
        self.network_obj = network_obj
        self.limits_obj = limits_obj

        self.buy_orders = []
        self.sell_orders = []

        load_eans = self.network_obj.net.load["ean"].values
        self._ean_to_index = {ean: idx for idx, ean in enumerate(load_eans)}

        #Indexes for the order list
        self.id_index = 0
        self.ean_index = 1
        self.order_type_index = 2
        self.amount_index = 3
        self.price_index = 4
        self.time_index = 5

        # For statistics
        self.total_buys = 0
        self.matched_buys = 0
        self.traded_limits = []
        self.id_buy_orders = 0
        self.id_sell_orders = 0

    def create_order(self, customer, order_type, amount, price=0.1, time=-1):
        price = round(price, 5)
        amount = round(amount, 3)
        # customer_ean = customer['ean']
        customer_ean = customer['ean']
        if order_type == "BUY":
            self.buy_orders.append([self.id_buy_orders, customer_ean, order_type, amount, price, time])
            self.id_buy_orders += 1
        elif order_type == "SELL":
            self.sell_orders.append([self.id_sell_orders, customer_ean, order_type, amount, price, time])
            self.id_sell_orders += 1
        # print(f"#Creating {order_type} order for customer {customer_ean}: {amount:.2f}kW at {price}$ for {time} timesteps")

    def submit_orders(self):
        #This does not allow for more than one request: either a BUY or SELL, not both
        #TODO: fix this if needed
        minimum_order_request = 0.050 #kW
        for idx, row in self.network_obj.net.load.iterrows(): 
            energy_usage = row['p_mw'] * 1000
            # print(f"{energy_usage:.3f}", limits_obj.limits[idx,:])
            if(energy_usage > 0): #Consumption
                limit = self.limits_obj.limits[idx, 1]
                diff = limit - energy_usage

                if(abs(diff) > minimum_order_request):
                    if diff > 0:
                        price = 0.1 * np.random.rand()
                        self.create_order(row, "SELL", diff, price)
                    else:
                        price = 0.3 * np.random.rand()
                        self.create_order(row, "BUY", abs(diff), price)
                        self.total_buys += 1
            else: #Generation
                limit = self.limits_obj.limits[idx, 0]
                diff = limit - energy_usage

                if(abs(diff) > minimum_order_request):
                    if diff < 0:
                        price = 0.1 * np.random.rand()
                        self.create_order(row, "SELL", diff, price)
                    else:
                        price = 0.3 * np.random.rand()
                        self.create_order(row, "BUY", diff, price)
                        self.total_buys += 1
            # print()

    def orderCompatibility_check(self, buy_order, sell_order):
        """
        Check if a buyer can match with a seller
        Return True if compatible, false otherwise
        """
        buy_price = buy_order[self.price_index]  # max price that the buyer wants to pay
        sell_price = sell_order[self.price_index]  # min price that the seller wants to sell

        buy_sign = np.sign(buy_order[self.amount_index])
        sell_sign = np.sign(sell_order[self.amount_index])

        if sell_price <= buy_price and buy_sign == sell_sign:
            return True
        return False

    def set_price(self, buy_order, sell_order):
        """
        Set the exchanges price (seller price)
        """
        transaction_price = sell_order[self.price_index]
        return transaction_price

    def compute_social_welfare(self, buy_order, sell_order, traded_limit):
        """
        Compute the social welfare of a transaction between a buyer and a seller.
        """
        buy_price = buy_order[self.price_index]  # Price max of buyer
        sell_price = sell_order[self.price_index]  # Price min of seller

        # Gains économiques
        buyer_surplus = (buy_price - sell_price) * traded_limit  # buyer benefit
        seller_surplus = traded_limit * sell_price  # seller benefit

        social_welfare = buyer_surplus + seller_surplus  # benefit sum
        return social_welfare

    def determine_trade_limits(self, buy_order, sell_order):
        """
        Determine the trade limits between a buyer and a seller
        """
        limits = self.limits_obj.limits

        buyer_index = self._ean_to_index[buy_order[self.ean_index]]
        seller_index = self._ean_to_index[sell_order[self.ean_index]]

        buyer_L = buy_order[self.amount_index]
        seller_L = sell_order[self.amount_index]

        sign = np.sign(buyer_L)
        index = int(sign == 1)

        iterations = 5
        L_after = limits.copy()
        traded_limit = min(buyer_L, seller_L)
        for i in range(iterations):
            L_after[buyer_index, index] = limits[buyer_index, index] + sign * traded_limit
            L_after[seller_index, index] = limits[seller_index, index] - sign * traded_limit
            if self.limits_obj.safety_verification(L_after):
                break
            traded_limit *= 0.8
        # print(f"Trade limit determined: {traded_limit:.4f}. {i+1} iterations. Buyer asked {buyer_L:.4f} and Seller asked {seller_L:.4f}")
        return traded_limit, L_after
        

    def update_orders(self, buy_order, sell_order, traded_limit):
        self.buy_orders.remove(buy_order)
        self.sell_orders.remove(sell_order)
        # self.buy_orders.append((buy_order[0], buy_order[1] - traded_limit, buy_order[2], buy_order[3]))
        # self.sell_orders.append((sell_order[0], sell_order[1] - traded_limit, sell_order[2], sell_order[3]))
        
    def market_clearing(self):
        """
        Optimize Exchanges between sellers and buyers while respecting the DOEs and the network safety
        """
        # start_time = time.time()
        traded_limits = []
        for buy_order in self.buy_orders.copy():
            # buy_loop_start_time = time.time()
            buyer_id, buyer_ean, _, buyer_L, buyer_price, buyer_time = buy_order
            valid_trades = []

            compatible_sell_orders = [
                sell_order
                for sell_order in self.sell_orders
                if self.orderCompatibility_check(buy_order, sell_order)
            ]
            
            # determine_trade_limits_time = 0
            # sell_loop_start_time = time.time()
            for sell_order in compatible_sell_orders:
                seller_id, seller_ean, _, seller_L, seller_price, seller_time = sell_order

                price = self.set_price(buy_order, sell_order)

                # trade_limits_start_time = time.time()
                traded_limit, L_after = self.determine_trade_limits(buy_order, sell_order)
                # trade_limits_end_time = time.time()
                # determine_trade_limits_time += trade_limits_end_time - trade_limits_start_time

                social_welfare = self.compute_social_welfare(buy_order, sell_order, traded_limit)

                valid_trades.append({
                    "buyer": buyer_ean,
                    "buyer_id": buy_order[self.id_index],
                    "seller": seller_ean,
                    "seller_id": sell_order[self.id_index],
                    "price": price,
                    "traded_limit": traded_limit,
                    "social_welfare": social_welfare,
                    "time": buyer_time,
                    "limits_after": L_after
                })
                # print(f"Transaction candidate: Buyer {buyer_ean}, {buyer_L} kW. Seller {seller_ean}, {seller_L} kW. Price {price} EUR/kW. Social welfare {social_welfare} EUR")
                # sell_loop_end_time = time.time()
            # print(f" \t - Inner sell order loop iteration took: {sell_loop_end_time - sell_loop_start_time:.4f} seconds. Trade limits took: {determine_trade_limits_time:.4f} seconds")

            if valid_trades:
                best_trade = max(valid_trades, key=lambda x: x["social_welfare"])
                # print(f"Transaction performed: Buyer {buyer_ean}, {buyer_L} kW. Seller {best_trade['seller']}. Exchanged {best_trade['traded_limit']} kW @ {best_trade['price']:.3f} EUR/kW")
                # print()
                sell_order = [s for s in self.sell_orders if s[0] == best_trade['seller_id']][0]
                self.update_orders(buy_order, sell_order, best_trade["traded_limit"])
                
                self.limits_obj.limits = best_trade["limits_after"]

                self.matched_buys += 1
                traded_limits.append(best_trade["traded_limit"])
            # buy_loop_end_time = time.time()
            # print(f" - Entire buy order loop took: {buy_loop_end_time - buy_loop_start_time:.4f} seconds")
        self.traded_limits.append(traded_limits)
        # end_time = time.time()
        # print(f"Market clearing completed in {end_time - start_time:.2f} seconds")
        # print()