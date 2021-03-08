# Syntax

# Param 1 and Param 2 are parameters that python expects you to pass
# once you create an instance of this object


class NameOfClass:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def some_method(self):
        # perform some action
        print(self.param1)


# Class keyword and creating attributes
print(type(NameOfClass))


# An attribute is a characteristic of an object
class Dog:
    # init is the constructor for the class
    def __init__(self, breed, name, spots):
        # Attributes
        # We take in the argument
        # Assign it using self.attribute_name
        self.my_attribute = breed
        self.name = name
        self.spots = spots


# instance of the dog class
my_dog = Dog(breed="Lab", name="Tommy", spots=False)
print(type(my_dog))

print(my_dog.my_attribute)
print(my_dog.name)
print(my_dog.spots)


# Class object attributes


class secondDog:
    # Class object attributes
    # Same for any instance of a class

    species = "mammal"

    def __init__(self, breed, name, spots):
        self.breed = breed
        self.name = name
        self.spots = spots

    # Methods - operations/actions

    def bark(self, owner):
        print(f"WOOF! My name id {self.name} and my owner is {owner}.")


second_dog = secondDog(
    breed="African",
    name="Sting",
    spots=True,
)
# Calling attributes
print(second_dog.breed)
print(second_dog.spots)
print(second_dog.species)

# Calling methods
print(second_dog.bark("Charles"))

# CIRCLE CLASS


class Circle:
    # Class object attribute

    pi = 3.14

    def __init__(self, radius=1):
        self.radius = radius
        # attributes don't have to be defined in the parameters
        self.area = self.pi * radius ** 2

    # Method

    def get_circumference(self):
        #Circle.pi referencing class object attribute or self.pi
        return self.radius * Circle.pi * 2


my_circle = Circle(radius=30)

print(my_circle.pi)
print(my_circle.radius)
print(my_circle.get_circumference())
print(my_circle.area)
