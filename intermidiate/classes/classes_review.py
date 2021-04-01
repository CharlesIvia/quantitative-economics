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
        # Circle.pi referencing class object attribute or self.pi
        return self.radius * Circle.pi * 2


my_circle = Circle(radius=30)

print(my_circle.pi)
print(my_circle.radius)
print(my_circle.get_circumference())
print(my_circle.area)

# Inheritance in classes

# Base class


class Animal(object):
    def __init__(self, age):
        self.age = age
        self.name = None

    def get_age(self):
        return self.age

    def get_name(self):
        return self.name

    def set_age(self, newage):
        self.age = newage

    def set_name(self, newname=""):
        self.name = newname

    def __str__(self):
        return "animal: " + str(self.name) + ": " + str(self.age)


# Derived class


class Person(Animal):
    def __init__(self, name, age):
        Animal.__init__(self, age)  # Call Animal constructor
        self.set_name(name)  # Call Animal's method
        self.friends = []  # add new data attribute

    def get_friends(self):
        return self.friends

    # new methods
    def add_friend(self, fname):
        if fname not in self.friends:
            self.friends.append(fname)

    def speak(self):
        print("hello")

    def age_diff(self, other):
        diff = self.age - other.age
        print(abs(diff), "year difference.")

    # override Animal's __str__ method
    def __str__(self):
        return "person: " + str(self.name) + ": " + str(self.age)


gats = Person("Gats", 22)
print(gats)

ethan = Person("Ethan", 8)

gats.add_friend("Gichobi")
gats.add_friend("Kithuu")
gats.add_friend("Kamau")

print(gats.get_friends())
print(gats.speak)
print(gats.age_diff(ethan))
