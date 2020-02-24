def sarima_tuning(data, range_max, diff_times):
    
    p = d = q = range(0, range_max)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], diff_times) for x in list(itertools.product(p, d, q))]
    
    #give a random aic value
    min_aic = 999999999

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(data,
                                                  order=param,
                                                  seasonal_order=param_seasonal,
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False
                                                 )

                results = model.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

                #find the lowest AIC
                if results.aic < min_aic:
                    min_aic = results.aic
                    min_aic_model = results
    
            except:
                continue
                
    return min_aic_model