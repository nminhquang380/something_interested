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
Encapsulation is one of the key features of object-oriented programming. Encapsulation refers to the bundling of attributes and methods inside a single class.

It prevents outer classes from accessing and changing attributes and methods of a class. This also helps to achieve data hiding.

In Python, we denote private attributes using underscore as the prefix i.e single _ or double __. For example:
```
class Computer:

    def __init__(self):
        self.__maxprice = 900

    def sell(self):
        print("Selling Price: {}".format(self.__maxprice))

    def setMaxPrice(self, price):
        self.__maxprice = price

c = Computer()
c.sell()

# change the price
c.__maxprice = 1000
c.sell()

# using setter function
c.setMaxPrice(1000)
c.sell()

# Selling Price: 900
# Selling Price: 900
# Selling Price: 1000
```

In the above program, we defined a Computer class.

We used __init__() method to store the maximum selling price of Computer. Here, notice the code `c.__maxprice = 1000`

Here, we have tried to modify the value of `__maxprice` outside of the class. However, since `__maxprice` is a private variable, this modification is not seen on the output.

As shown, to change the value, we have to use a setter function i.e `setMaxPrice()` which takes price as a parameter.

## Polymorphism
Polymorphism is another important concept of object-oriented programming. It simply means more than one form.

That is, the same entity (method or operator or object) can perform different operations in different scenarios. Let's see an example:

```
class Polygon:
    # method to render a shape
    def render(self):
        print("Rendering Polygon...")

class Square(Polygon):
    # renders Square
    def render(self):
        print("Rendering Square...")

class Circle(Polygon):
    # renders circle
    def render(self):
        print("Rendering Circle...")
    
# create an object of Square
s1 = Square()
s1.render()

# create an object of Circle
c1 = Circle()
c1.render()

# Rendering Square...
# Rendering Circle...
```

In the above example, we have created a superclass: `Polygon` and two subclasses: `Square` and `Circle`. Notice the use of the `render()` method.

The main purpose of the `render()` method is to render the shape. However, the process of rendering a square is different from the process of rendering a circle.

Hence, the `render()` method behaves differently in different classes. Or, we can say render() is polymorphic.