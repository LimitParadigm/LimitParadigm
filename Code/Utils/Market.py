import copy

class Market:
    def __init__(self, network_obj, limits_obj):
        self.network_obj = network_obj
        self.limits_obj = limits_obj

        self.buy_orders = []
        self.sell_orders = []


    def create_order(self, customer, order_type, amount, price=0.1, time=-1):
        # customer_ean = customer['ean']
        customer_ean = customer['ean']
        if order_type == "BUY":
            self.buy_orders.append([customer_ean, order_type, amount, price, time])
        elif order_type == "SELL":
            self.sell_orders.append([customer_ean, order_type, amount, price, time])
        print(f"#Creating {order_type} order for customer {customer_ean}: {amount:.2f}kW at {price}$ for {time} timesteps")

    def submit_orders(self):
        #This does not allow for more than one request: either a BUY or SELL, not both
        #TODO: fix this if needed
        minimum_order_request = 0.050 #kW
        price = 0.001
        for idx, row in self.network_obj.net.load.iterrows(): 
            energy_usage = row['p_mw'] * 1000
            # print(f"{energy_usage:.3f}", limits_obj.limits[idx,:])
            if(energy_usage > 0): #Consumption
                limit = self.limits_obj.limits[idx, 1]
                diff = limit - energy_usage
                diff = round(diff, 2)

                if(abs(diff) > minimum_order_request):
                    if diff > 0:
                        self.create_order(row, "SELL", diff, price)
                    else:
                        self.create_order(row, "BUY", abs(diff), 2*price)
            else: #Generation
                limit = self.limits_obj.limits[idx, 0]
                diff = limit - energy_usage

                if(abs(diff) > minimum_order_request):
                    if diff < 0:
                        self.create_order(row, "SELL", diff, price)
                    else:
                        self.create_order(row, "BUY", diff, 2*price)
            # print()

    def orderCompatibility_check(self, buy_order, sell_order):
        """
        Check if a buyer can match with a seller
        Return True if compatible, false otherwise
        """
        buy_price = buy_order[3]  # max price that the buyer wants to pay
        sell_price = sell_order[3]  # min price that the seller wants to sell

        if sell_price <= buy_price:
            return True
        return False

    def set_price(self, buy_order, sell_order):
        """
        Set the exchanges price (seller price)
        """
        transaction_price = sell_order[3]
        return transaction_price

    def compute_social_welfare(self, buy_order, sell_order, traded_limit):
        """
        Compute the social welfare of a transaction between a buyer and a seller.
        """
        buy_price = buy_order[3]  # Price max of buyer
        sell_price = sell_order[3]  # Price min of seller

        # Gains économiques
        buyer_surplus = (buy_price - sell_price) * traded_limit  # buyer benefit
        seller_surplus = traded_limit * sell_price  # seller benefit

        social_welfare = buyer_surplus + seller_surplus  # benefit sum
        return social_welfare

    def determine_trade_limits(self, buy_order, sell_order):
        """
        Determine the trade limits between a buyer and a seller
        """
        buyer_index = self.network_obj.net.load[self.network_obj.net.load["ean"] == buy_order[0]].index[0]
        seller_index = self.network_obj.net.load[self.network_obj.net.load["ean"] == sell_order[0]].index[0]
        
        buyer_L = buy_order[2]
        seller_L = sell_order[2]

        sign = 1 if buyer_L > 0 else -1

        iterations = 10
        L_after = copy.deepcopy(self.limits_obj.limits)
        traded_limit = min(buyer_L, seller_L)
        for i in range(iterations):
            L_after[buyer_index] = self.limits_obj.limits[buyer_index] + sign * traded_limit
            L_after[seller_index] = self.limits_obj.limits[seller_index] - sign * traded_limit
            if self.limits_obj.safety_verification(L_after):
                break
            traded_limit *= 0.9
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
        valid_trades = []

        for buy_order in self.buy_orders:
            buyer_ean, _, buyer_L, buyer_price, buyer_time = buy_order

            for sell_order in self.sell_orders:
                seller_ean, _, seller_L, seller_price, seller_time = sell_order

                if self.orderCompatibility_check(buy_order, sell_order):
                    price = self.set_price(buy_order, sell_order)
                    traded_limit, L_after = self.determine_trade_limits(buy_order, sell_order)
                    social_welfare = self.compute_social_welfare(buy_order, sell_order, buyer_L)
                    valid_trades.append({
                        "buyer": buyer_ean,
                        "seller": seller_ean,
                        "price": price,
                        "traded_limit": traded_limit,
                        "social_welfare": social_welfare,
                        "time": buyer_time
                    })
            if valid_trades:
                best_trade = max(valid_trades, key=lambda x: x["social_welfare"])
                print(f"Transaction performed: Buyer {buyer_ean}, {buyer_L} kW. Seller {best_trade['seller']}. Exchanged {best_trade['traded_limit']} kW @ {best_trade['price']:.3f} EUR/kW")
                self.update_orders(buy_order, sell_order, best_trade["traded_limit"])
                
                self.limits_obj.limits = L_after