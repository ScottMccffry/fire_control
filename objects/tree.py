class Tree:
    def __init__(self, height, width, tree_type, humidity, combustion_coefficient):
        self.height = height
        self.width = width
        self.tree_type = tree_type
        self.humidity = humidity
        self.combustion_coefficient = combustion_coefficient

# Example usage:
# tree = Tree(10, 5, "Oak", 80)
# print(tree.height)  # Output: 10
# print(tree.width)  # Output: 5
# print(tree.tree_type)  # Output: Oak
# print(tree.humidity)  # Output: 80