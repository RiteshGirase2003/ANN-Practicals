def mcculloch_pitts(inputs, weights):
    """McCulloch-Pitts neuron model."""
    net_input = sum(x * w for x, w in zip(inputs, weights))
    output = 1 if net_input >= 1 else 0
    return output

# Define the situations and inputs
situations = [
    ("Not raining, not sunny", [0, 0]),
    ("Not raining, sunny", [0, 1]),
    ("Raining, not sunny", [1, 0]),
    ("Raining, sunny", [1, 1])
]

# Define weights for the inputs (X1, X2)
weights = [1, 1]

# Test the McCulloch-Pitts neuron for each situation
for situation, inputs in situations:
    output = mcculloch_pitts(inputs, weights)
    carry_umbrella = "needs to" if output == 1 else "does not need to"
    print(f"In situation '{situation}', John {carry_umbrella} carry an umbrella.")
