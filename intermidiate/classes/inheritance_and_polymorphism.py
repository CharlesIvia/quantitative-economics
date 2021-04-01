# INHERITANCE AND POLYMORPHISM

# Inheritance is a way to form new classes using classes that have already been defined

# Benefits: code reusability and abstracting complexity

# Base class
class Animal:
    def __init__(self):
        print("Animal Created")

    def who_am_i(self):
        print("I am an Animal.")

    def eat(self):
        print("I am eating!")


my_animal = Animal()  # instance of base class Animal
print(my_animal)

print(my_animal.who_am_i())
print(my_animal.eat())

# The Cat class  - derived class


class Cat(Animal):
    def __init__(self):
        Animal.__init__(self)
        print("Cat created!")


my_cat = Cat()
print(my_cat)
print(my_cat.who_am_i())


class Cat(Animal):
    def __init__(self):
        Animal.__init__(self)
        print("Cat created!")

    # Overwrite methods from animal

    def who_am_i(self):
        print("I am a cat!")

    # Add methods
    def meow(self):
        print("Meaaaooow!")


my_cat = Cat()
print(my_cat)
print(my_cat.who_am_i())
print(my_cat.meow())

# POLYMORPHISM


class Dog:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return self.name + " says woof!"


class Cat:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return self.name + " says meaow!"


niko = Dog("niko")
felix = Cat("felix")

# Demonstrating polymorphism (cat and dog class share same method name speak)

print(niko.speak())
print(felix.speak())

for pet in [niko, felix]:
    print(type(pet))
    print(pet.speak())


def pet_speak(pet):
    print(pet.speak())


pet_speak(niko)
pet_speak(felix)

# ABSTRACT CLASSES AND INHERITANCE

# Abstract classes never expect to be instantiated
# Designed to serve as a base class


class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement this abstract method")


class Dog(Animal):
    def speak(self):
        return self.name + " says woof!"


class Cat(Animal):
    def speak(self):
        return self.name + " says meaow!"


fido = Dog("Fido")
isis = Dog("Isis")

print(fido.speak())
print(isis.speak())


# MAGIC/DUNDER METHODS


class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.autor = author
        self.pages = pages

    def __str__(self):
        return f"{self.title} by {self.autor}"

    def __len__(self):
        return self.pages

    def _del_(self):
        print("A book object has been deleted!")


b = Book("Pythonista", "Charles Ivia", 674)
print(b)

print(len(b))
