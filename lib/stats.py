import csv

def write_stats(samples, features, loss, epochs, lr, batch_size, cross_k, mse_train, mse_test, mae_train, mae_test, residual_train, residual_test):
    with open('../data/stats.csv', 'a') as f:
        newrow = [samples, features, loss, epochs, lr, batch_size, cross_k, mse_train, mse_test, mae_train, mae_test, residual_train, residual_test]
        writer = csv.writer(f)
        writer.writerow(newrow)
