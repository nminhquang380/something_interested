# OOP with Python

Python is a versatie programming language that support varous programming styles, including object-oriented programming (OOP) through the use of objects and classes.

An object is any entity that has attributes and behaviors. For example, a `parrot` is an object. It has
- attributes - name, age, color, etc.
- behavior - dancing, singing, etc.
Similarly, a class is a blueprint for that object.

## Python Class and Object
```python
class Parrot:

    # class attribute
    name = ""
    age = 0

# create parrot1 object
parrot1 = Parrot()
parrot1.name = "Blu"
parrot1.age = 10

# create another object parrot2
parrot2 = Parrot()
parrot2.name = "Woo"
parrot2.age = 15

# access attributes
print(f"{parrot1.name} is {parrot1.age} years old")
print(f"{parrot2.name} is {parrot2.age} years old")
```

In the above example, we created a class with the name `Parrot` with two attributes: `name` and `age`.

Then, we create instances of the `Parrot` class. Here, parrot1 and parrot2 are references (value) to our new objects.

We then accessed and assigned different values to the instance attributes using the objects name and the `.` notation.

## Python Inheritance

Inheritance is a way of creating a new class for using details of an existing class without modifying it.

The new formed class is a derived class (or child class). Similarly, the existing class a base class (or parent class).

### Syntax
Here's the syntax of the inheritance in Python:

```
# define a superclass
class super_class:
    # attributes and method definition

# inheritance
class sub_class(super_class):
    # attributes and method of super_class
    # attributes and method of sub_class
```
### is-a relationship
In Python, inheritance is an is-a relationship. That is, we use inheritance only if there exists an is-a relationship between two classes. For example:
1. Car is a Vehicle
2. Apple is a Fruit
3. Cat is an Animal
Here, Car can inherit from Vehicle, Apple can inherit from Fruit, and so on.

### Method Overriding in Python Inheritance.

Example:
```
class Animal:

    # attributes and method of the parent class
    name = ""
    
    def eat(self):
        print("I can eat")

# inherit from Animal
class Dog(Animal):

    # override eat() method
    def eat(self):
        print("I like to eat bones")

# create an object of the subclass
labrador = Dog()

# call the eat() method on the labrador object
labrador.eat()
```

In the above example, the same method `eat()` is present in both the `Dog` class and the `Animal` class.

Now, when we call the `eat()` method using the object of the `Dog` subclass, the method of the `Dog` class is called.

This is because the `eat()` method of the `Dog` subclass overrides the same method of the `Animal` superclass.

### The super() Method 

Previously we saw that the same method in the subclass overrides the method in the superclass.

However, if we need to access the superclass method from the subclass, we use the super() method. For example:
```
class Animal:

    name = ""
    
    def eat(self):
        print("I can eat")

# inherit from Animal
class Dog(Animal):
    
    # override eat() method
    def eat(self):
        
        # call the eat() method of the superclass using super()
        super().eat()
        
        print("I like to eat bones")

# create an object of the subclass
labrador = Dog()

labrador.eat()
```

## Python Encapsulation
