def make_table(model_name, first_column, second_column, first_train, second_train, first_valid, second_valid, first_test, second_test):
    print(f"Wyniki dla {model_name}")
    print(f"{'Zbiór':<12}{first_column:>10}  {second_column:>10}")
    print(f"{'Train':<12}{first_train:>10.4f}  {second_train:>10.4f}")
    print(f"{'Validation':<12}{first_valid:>10.4f}  {second_valid:>10.4f}")
    print(f"{'Test':<12}{first_test:>10.4f}  {second_test:>10.4f}")