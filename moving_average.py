def moving_average_algorithm(factor: int, data: []):
    predicted_values = []
    for i in range(len(data)):
        if i <= factor:
            predicted_values.append(data[i])
        else:
            average_value: float = sum(data[i-factor:i]) / factor
            predicted_values.append(average_value)

    return predicted_values
