# HybridQ-Decorators: Useful decorators to extend object functionalities

**HybridQ-Decorators** is a collection of useful decorators to extend the
functionality of objects.

## Installation

**HybridQ-Decorators** can be installed as stand-alone library using `pip`:
```
pip install 'git+https://github.com/nasa/hybridq#egg=hybridq-decorators&subdirectory=hybridq/modules/hybridq_decorators'
```

## Getting Started

Tutorials on how to use **HybridQ-Decorators** can be found in
[hybridq-decorators/tutorials](https://github.com/nasa/hybridq/tree/main/hybridq/modules/hybridq_decorators/tutorials).

## How to Use

**HybridQ-Decorators** is a decorator libraries to enhance objects easily.  For
instance, **HybridQ-Decorators** extend the decorator `property` to classes:
```
from hybridq_decorators import classproperty, ClassProperty


class A(ClassProperty):
    _x = 0

    @classproperty
    def x(cls):
        return cls._x


# Class properties can be directly access before ...
assert (A.x == 0)

# ... and after instantiating the object
assert (A().x == 0)

# As 'property', 'classproperty' is read only:
try:
    A.x = 1
except AttributeError as e:
    print(e)
# > can't set attribute 'x'
```
`classproperty` uses the same syntax as `property`:
```
class A(ClassProperty):
    _x = 0

    @classproperty
    def x(cls):
        print("Getter")
        return cls._x

    @x.setter
    def x(cls, x):
        print("Setter")
        cls._x = int(x)

    @x.deleter
    def x(cls):
        print("Deleter")
        del cls._x


assert (A.x == 0)
# > Getter
A.x = 2.2
# > Setter
assert (A.x == 2)
# > Getter
del A.x
# > Deleter

try:
    A.x
except AttributeError as e:
    print(e)
# > Getter
# > type object 'A' has no attribute '_x'
```
HybridQ-Decorators also provides `staticvars`. `staticvars` are simplified
`classproperty`s that allow to add read-only attributes to a class:
```
from hybridq_decorators import staticvars


@staticvars(x=1)
class A(ClassProperty):
    ...


assert (A.x == 1)
try:
    A.x = 2
except AttributeError as e:
    print(e)
# > can't set attribute 'x'

# Static vars cannot be changed while dynamically creating new
# types, unless the static variable is declared 'mutable'

try:
    type('B', (A, ), {}, static_vars=dict(x=2))
except AttributeError as e:
    print(e)
# > can't set attribute 'x'


@staticvars(x=1, mutable=True)
class A(ClassProperty):
    ...


# Dynamically create a new type
B = type('B', (A, ), {}, static_vars=dict(x=2))

# Check
assert (B.x == 2)
```
Static variables can also be declared without specifying a value:
```
@staticvars('x,y', mutable=True)
class A(ClassProperty):
    ...


try:
    A.x
except AttributeError as e:
    print(e)
# > type object 'A' has no attribute 'x'

try:
    A().y
except AttributeError as e:
    print(e)
# > type object 'A' has no attribute '__A_static_y'
```
This is useful for instance when new types are dynamically created,
each one with different values of static vars:
```
B1 = type('B1', (A, ), {}, static_vars=dict(x=1, y=2))
B2 = type('B1', (A, ), {}, static_vars=dict(x=3, y=4))

assert (B1.x == 1 and B1.y == 2)
assert (B2.x == 3 and B2.y == 4)

# A mix of mutable and non mutable variables can be declared
# using multiple 'staticvars'


@staticvars('x', mutable=True)
@staticvars(y=3)
class A(ClassProperty):
    ...


B1 = type('B1', (A, ), {}, static_vars=dict(x=42))
assert (B1.x == 42)
assert (B1.y == 3)

# B1.y cannot be changed
try:
    B1 = type('B1', (A, ), {}, static_vars=dict(y=42))
except AttributeError as e:
    print(e)
# > can't set attribute 'y'
```
`staticvars` also allow to transform and check static
variables. This is useful for mutable static vars:
```
@staticvars(x='2',
            transform=dict(x=lambda x: float(x)**2),
            check=dict(x=lambda x: x > 3),
            mutable=True)
class A(ClassProperty):
    ...


assert (isinstance(A.x, float) and A.x == 4)

# Create a new type
B1 = type('B1', (A, ), {}, static_vars=dict(x='3.32'))

# B1.x should be a float
assert (isinstance(B1.x, float) and B1.x == 3.32**2)

# If a check fail, a 'ValueError' is raised
try:
    type('B1', (A, ), {}, static_vars=dict(x='1.32'))
except ValueError as e:
    print(e)
# > Check failed for variable 'x'
```
Similarly to `staticvars`, **HybridQ-Decorators** provide a simple way to
declare class attributes using `attributes`. The decorator `attributes` allows
to declare attributes with both default and not default values:
```
from hybridq_decorators import attributes


# 'a' and 'b' have default values
@attributes('c', a=1, b=2)
class A:
    ...


# A 'TypeError' will be raised if missing required attributes are
# missing
try:
    A()
except TypeError as e:
    print(e)
# > A.__init__() missing required keyword-only argument: 'c'

# Get a new instance
a = A(c='hello')

# Check
assert (a.a == 1 and a.b == 2 and a.c == 'hello')

# Defaul values can be overwritten
a = A(a=-1, b=-2, c=-3)

# Check
assert (a.a == -1 and a.b == -2 and a.c == -3)
```
Similat to `staticvars`, `attributes` allows to transform and
check attributes:
```
@attributes(a=1.1, transform=dict(a=int), check=dict(a=lambda a: a < 2))
class A:
    ...


# Get new instance
a = A()

# a.a should be an int
assert (isinstance(a.a, int) and a.a == 1)

a = A(a='-3')
assert (isinstance(a.a, int) and a.a == -3)

# If check fails, a 'ValueError' is raised
try:
    a = A(a='5')
except ValueError as e:
    print(e)
# > Check failed for variable 'a'
```
The user can also provides a new `__init__` for the class. Attributes declared
using `attributes` are parsed before `__init__`:
```
@attributes('a', b=0)
class A:

    def __init__(self, c, d=1):
        self.c = c
        self.d = d
        print(f'{self.a=}')
        print(f'{self.b=}')
        print(f'{self.c=}')
        print(f'{self.d=}')


a = A(a=1, c='hello')
# > self.a=1
# > self.b=0
# > self.c='hello'
# > self.d=1

assert (a.a == 1 and a.b == 0 and a.c == 'hello' and a.d == 1)
```
When dealing with inheritance, it is useful to make sure that a child type
correctly implements the required members. The decorators `requires` and
`provides` allow the implementation of "virtual" members in Python:
```
from hybridq_decorators import requires, provides


@requires('a,b')
class A:
    ...


# Since 'a' and 'b' are required but 'A' implements none,
# raise an 'AttributeError'
try:
    A()
except AttributeError as e:
    print(e)
# > type object 'A' requires 'a'


# Members can be added in multiple ways
class B(A):
    a: int = 1
    b: float = 2.2


b = B()
assert (b.a == 1 and b.b == 2.2)


@attributes(a=1)
@staticvars(b=2)
class C(A, ClassProperty):
    ...


c = C()
assert (c.a == 1 and c.b == 2)


class D:

    def a(self):
        return 42

    @property
    def b(self):
        return 'hello!'


d = D()
assert (d.a() == 42 and d.b == 'hello!')


class E(A):
    a: str = '42'


try:
    E()
except AttributeError as e:
    print(e)
# > type object 'E' requires 'b'


@attributes('b')
class F(E):
    ...


f = F(b=0)
assert (f.a == '42' and f.b == 0)


class G(A):

    def __init__(self, a, b=0):
        self.a = a
        self.b = b


g = G(a=1)
assert (g.a == 1 and g.b == 0)


class H(A):
    __slots__ = ('a', 'b')

    def __init__(self, a, b):
        self.a = a
        self.b = b


h = H(a=1, b=2)
assert (h.a == 1 and h.b == 2)
```
In some cases, it may be useful to override a required member using `provides`:
```
@requires('a,b')
class A:
    ...


@provides('b')
class B(A):
    a: int = 42


b = B()
assert (b.a == 42)
try:
    b.b
except AttributeError as e:
    print(e)
```
**HybridQ-Decorators** also provides `compare` to simplify the comparison
between objects:
```
from hybridq_decorators import compare


# Compare only 'a'
@compare('a')
@attributes('a,b')
class A:
    ...


assert (A(a=1, b=2) == A(a=1, b=5))
assert (A(a=1, b='hello') != A(a=2, b='hello'))

# It is also possible to provide a user-define function to compare
# methods:


@compare('a', b=lambda x, y: abs(x) == abs(y))
@attributes('a,b')
class A:
    ...


assert (A(a=1, b=2) == A(a=1, b=-2))
assert (A(a=1, b=2) == A(a=1, b=2))

# Moreover, if '__eq__' is present, '__eq__' is checked as well


def _eq1(x, y):
    print('in _eq1')
    return True


def _eq2(x, y):
    print('in _eq2')
    return True


@attributes(a=0)
@compare(a=_eq1)
class A:
    ...


@compare(a=_eq2)
class B(A):

    def __eq__(self, other):
        print('in __eq__')
        return super().__eq__(other) and True


assert(A() == B())
# > in __eq__
# > in _eq1
# > in _eq2
```
User-defined objects with local references cannot be easily pickled.  To
overcome this, **HybridQ-Decorators** provides the decorator `pickler` to
enable `pickle` for arbitrary objects:
```
from hybridq_decorators import pickler, Pickler
import pickle


class A:
    ...


a = A()
a.x = lambda: 1

try:
    pickle.dumps(a)
except pickle.PicklingError as e:
    print(e)
# > Can't pickle <function <lambda> at 0x7f91e819d510>: attribute lookup <lambda> on __main__ failed


# Any valid alternative for pickle can be provided
@pickler('cloudpickle')
class B(Pickler):
    ...


b = B()
b.x = lambda: 1

assert (pickle.loads(pickle.dumps(b)).x() == 1)
```
Finally, **HybridQ-Decorators** includes the decorator `printer` to control how
an object is printed:
```
from hybridq_decorators import printer, Printer

# The simplest way to control how a member of an object is printed
# is to provide a string:


@attributes(a=0)
@printer(a='a={self.a}')
class A(Printer):
    ...


# The output of 'a' is then printer as 'a={self.a}'.format(self)
print(A())
# > A(a=0)


# Otherwise, a callable that accepts self can be used
@attributes(a=0, b='42')
@printer(a='a={self.a}', b=lambda self: f'b={float(self.b)/3.1416:1.2f}π')
class A(Printer):
    ...


print(A())
# > A(a=0, b=13.37π)
```
A more finer control can be achieved using `PrintObject`:
```
from hybridq_decorators import PrintObject

# 'PrintObject' accepts 4 parameters:
# fn: callable | str
#     Function used to format `self`. If `callable`, it must accept a single
#     argument, `self`, and return an object convertible to `str`. String
#     will instead be evaluated using `str.format(self)`.
# pos: 'pre' | 'name' | 'bulk' | 'post', optional
#     Where to locate the argument. The output of `str(obj`) is divided in
#     four parts:
#                         [pre][name]([bulk])[post]
#     All four parts are processed in the same way, and then placed in the
#     right position.
# sep: str, optional
#     If this argument is followed by another within the same `pos`, use `sep`
#     to separate them.
# order: int, optional
#     Arguments are ordered accordingly to `order`.


@printer(
    name=PrintObject(f'HELLO', pos='name'),  # Change the name of the object
    prefix=PrintObject(f'PREFIX ',
                       pos='pre'),  # Add a prefix in front of the object
    postfix=PrintObject(f' POSTFIX',
                        pos='post'),  # Add a postfix at the end of the object
    a=PrintObject(' ** {self.a}', sep='', pos='post', order=10),
    b=PrintObject(lambda self: f'b={float(self.b)/3.1416:1.2f}π'))
@staticvars(a=32, b=16)
class A(Printer, ClassProperty):
    ...


print(A())
# > PREFIX HELLO(b=5.09π) ** 32 POSTFIX
```


## How To Cite

[1] S. Mandrà, J. Marshall, E. Rieffel, and R. Biswas, [*"HybridQ: A Hybrid
Simulator for Quantum Circuits"*](https://doi.org/10.1109/QCS54837.2021.00015),
IEEE/ACM Second International Workshop on Quantum Computing Software (QCS)
(2021)

## Licence

Copyright © 2021, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
