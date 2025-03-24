class Market:
    def __init__(self, network_obj, limits_obj):
        self.network_obj = network_obj
        self.limits_obj = limits_obj

        self.buy_orders = []
        self.sell_orders = []
        
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

    def create_order(self, customer, order_type, amount, price=0.1, time=-1):
        customer_ean = customer['ean']
        if order_type == "BUY":
            self.buy_orders.append({customer_ean, order_type, amount, price, time})
        elif order_type == "SELL":
            self.sell_orders.append({customer_ean, order_type, amount, price, time})
        print(f"#Creating {order_type} order for customer {customer_ean}: {amount:.2f}kW at {price}$ for {time} timesteps")