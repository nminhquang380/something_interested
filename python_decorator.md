# Decorators in Python

Decorators are very powerful and useful tool in Python since it allows programmers to modify
the behavior of a function or class. Decorators allow us to wrap another function in order to
extend the behaviour of the wrapped function, without permanently modifying it. But before diving
deep into decorators let us understand some concepts that will come in handy in learning the decorators.

## First Class Objects

In Python functions are first class objects which means **that functions** in python **can be used or passed as arguments**.
Properties of first class functions:
- A function is an instance of the object type.
- Can be stored in a variable.
- Can be passed the function as a parameter to another function.
- Can return the function from a function.
- Can store them in data structures such as hash tables, lists,...

**Ex1:** Treat function as objects.
```python
def shout(text):
    return text.upper()

print(shout('Hello'))

yell = shout

print(yell('Hello'))

# OUTPUT
# HELLO
# HELLO
```

In the above example, we have assigned the function shout to a variable. This will not call the fucntion, instead, it takes
the function object referenced by a `shout` and creates a second name pointing to it, `yell`.

**Ex2:** Passing the function as an argument
```python
def shout(text):
    return text.upper()

def whisper(text):
    return text.lower()

def greet(func):
    greeting = func('Hello')
    print(greeting)

# OUTPUT
# HELLO
# hello
```

**Ex3:** Returning functions from another function.
```python
def create_adder(x):
    def adder(y):
        return x+y
    return adder

add_15 = create_adder(15)
print(add_15(10))

# OUTPUT
# 25
```

## Decorators

As stated above the decorators are used to modify the behaviour of function or class. In Decorators, functions are taken as the argument into another function and then called inside the wrapper function.

**Decorator can modify** the behaviour:
```python
    # inner function can access the outer local
    # functions like in this case "func"
    def inner1():
        print("Hello, this is before function execution")
 
        # calling the actual function now
        # inside the wrapper function.
        func()
 
        print("This is after function execution")
         
    return inner1
 
 
# defining a function, to be called inside wrapper
def function_to_be_used():
    print("This is inside the function !!")
 
 
# passing 'function_to_be_used' inside the
# decorator to control its behaviour
function_to_be_used = hello_decorator(function_to_be_used)
 
 
# calling the function
function_to_be_used()

# OUTPUT
# Hello, this is before function execution
# This is inside the function !!
# This is after function execution
```

**Ex:** Find out the excution time of a function using a decorator.

```python
# importing libraries
import time
import math
 
# decorator to calculate duration
# taken by any function.
def calculate_time(func):
     
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
 
        # storing time before function execution
        begin = time.time()
         
        func(*args, **kwargs)
 
        # storing time after function execution
        end = time.time()
        print("Total time taken in : ", func.__name__, end - begin)
 
    return inner1
 
 
 
# this can be added to any function present,
# in this case to calculate a factorial
@calculate_time
def factorial(num):
 
    # sleep 2 seconds because it takes very less time
    # so that you can see the actual difference
    time.sleep(2)
    print(math.factorial(num))
 
# calling the function.
factorial(10)

# OUTPUT
# 3628800
# Total time taken in :  factorial 2.0061802864074707
```

### Chaining Decorators
In simpler terms chaining decorators means decorating a function with multiple decorators.

Example:
```python
# code for testing decorator chaining
def decor1(func):
    def inner():
        x = func()
        return x * x
    return inner
 
def decor(func):
    def inner():
        x = func()
        return 2 * x
    return inner
 
@decor1
@decor
def num():
    return 10
 
@decor
@decor1
def num2():
    return 10
   
print(num())
print(num2())

# OUTPUT
# 400
# 200
```

Reference: [Decorators in Python](https://www.geeksforgeeks.org/decorators-in-python/)