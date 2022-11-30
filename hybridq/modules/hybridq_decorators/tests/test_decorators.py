"""
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

Copyright © 2021, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The Decorama: Useful Decorators For Classes is licensed under the Apache
License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
from hybridq_decorators import *
import pytest


def test__classproperty():
    # Define a new class with the class property 'x'
    class A(ClassProperty):

        @classproperty
        def x(cls):
            return 1

    # Check value
    assert (A.x == 1 and A().x == 1)

    # 'x' should be read-only for 'A'
    try:
        A.x = 1
    except AttributeError as e:
        assert ("can't set attribute" in str(e))

    # 'x' should be read-only for instantiations of 'A'
    try:
        A().x = 1
    except AttributeError as e:
        assert ("can't set attribute" in str(e))

    class B(A):
        __y = None

        @classproperty
        def y(cls):
            """
            doc
            """
            return cls.__y

        @y.setter
        def y(cls, v):
            cls.__y = v

        @y.deleter
        def y(cls):
            del cls.__y

    # Check values
    assert (B.x == 1 and B().x == 1)

    # 'x' must be still read-only in 'B'
    try:
        B.x = 123
    except AttributeError as e:
        assert ("can't set attribute" in str(e))

    # Check default value
    assert (B.y is None and B().y is None)

    # Since 'setter' is set, it's now possible to assign values for 'y'
    B.y = 123

    # Check new value
    assert (B.y == 123 and B().y == 123)

    # Get a new object
    b = B()

    # The new value is propagated to new instances
    assert (b.y == 123)

    # It is possible to change the value for 'y' of instances
    b.y = 456

    # Check value
    assert (b.y == 456)

    # Howerver, the value for the original type is preserved
    assert (B.y == 123 and type(b).y == 123)

    # Since 'deleter' is provided, it can be used
    del b.y

    # Deleting 'y' for the instance will reset its value to the type value
    assert (b.y == 123)

    # Since 'deleter' is provided, it can be used
    del B.y

    # Accessing 'y' will now trigger an error
    try:
        B.y
    except AttributeError as e:
        assert (str(e) == "type object 'B' has no attribute '_B__y'")

    # Accessing 'y' will now trigger an error
    try:
        B().y
    except AttributeError as e:
        assert (str(e) == "'B' object has no attribute '_B__y'")


@pytest.mark.parametrize('pickler_name', ['dill', 'cloudpickle'])
def test__staticvars(pickler_name):
    import pickle

    # Cloudpickle is known to have problems
    if pickler_name == 'cloudpickle':
        pytest.skip("'cloudpickle' has issues with 'classproperty'")

    # Define a new class with different static variables
    @staticvars('a', transform=dict(a=float), mutable=True)
    @staticvars(b=42, c='hello!', check=dict(c=lambda c: isinstance(c, str)))
    class A(ClassProperty):
        pass

    # Check default values
    try:
        A.a
    except AttributeError as e:
        assert (str(e) == "type object 'A' has no attribute 'a'")
    assert (A.b == 42)
    assert (A.c == 'hello!')

    # 'a' is mutable but 'b' and 'c' are not
    type('A', (A,), {}, static_vars=dict(a=1.32))
    try:
        type('A', (A,), {}, static_vars=dict(b=1))
    except AttributeError as e:
        assert (str(e) == "can't set attribute 'b'")

    try:
        type('A', (A,), {}, static_vars=dict(c=1))
    except AttributeError as e:
        assert (str(e) == "can't set attribute 'c'")

    # 'a' must be convertible to 'float'
    try:
        B = type('B', (A,), {}, static_vars=dict(a='hello!', b=1, c='hi!'))
    except ValueError as e:
        assert (str(e) == "could not convert string to float: 'hello!'")

    # Define a new class with different static variables
    @staticvars('a', transform=dict(a=float), mutable=True)
    @staticvars(b=42,
                c='hello!',
                check=dict(c=lambda c: isinstance(c, str)),
                mutable=True)
    class A(ClassProperty):
        pass

    # 'c' must be a 'str'
    try:
        B = type(A)('B', (A,), {}, static_vars=dict(a=1.23, b=1, c=1))
    except ValueError as e:
        assert (str(e) == "Check failed for variable 'c'")

    # Get new type 'B' with different values for 'a', 'b' and 'c'
    B = type(A)('B', (A,), {}, static_vars=dict(b=1, c='hi!'))

    # Check new values
    try:
        B.a
    except AttributeError as e:
        assert (str(e) == "type object 'B' has no attribute 'a'")
    try:
        B().a
    except AttributeError as e:
        assert (str(e) == "type object 'B' has no attribute '__A_static_a'")
    assert (B.b == 1 and B().b == 1)
    assert (B.c == 'hi!' and B().c == 'hi!')

    # Get new type 'B' with different values for 'a', 'b' and 'c',
    # which is also pickeable
    B = pickler(pickler_name)(type('B', (A, Pickler), {},
                                   static_vars=dict(a='1.23',
                                                    b='hello!',
                                                    c='hi!')))

    # Get a new object
    b = B()

    # Dump/load using pickle
    _b = pickle.loads(pickle.dumps(b))

    # Check values
    assert (B.a == 1.23 and B().a == 1.23)
    assert (B.b == 'hello!' and B().b == 'hello!')
    assert (B.c == 'hi!' and B().c == 'hi!')
    assert (type(b).a == B.a)
    assert (type(b).b == B.b)
    assert (type(b).c == B.c)
    assert (type(_b).a == B.a)
    assert (type(_b).b == B.b)
    assert (type(_b).c == B.c)

    # All static variables must be read-only
    for k in 'abc':
        try:
            setattr(B, k, None)
        except AttributeError as e:
            assert ("can't set attribute" in str(e))

        try:
            setattr(type(_b), k, None)
        except AttributeError as e:
            assert ("can't set attribute" in str(e))

        try:
            setattr(_b, k, None)
        except AttributeError as e:
            assert ("can't set attribute" in str(e))

        try:
            setattr(b, k, None)
        except AttributeError as e:
            assert ("can't set attribute" in str(e))


@pytest.mark.parametrize('pickler_name', ['dill', 'cloudpickle'])
def test__requires(pickler_name):
    import importlib
    import pickle

    # Cloudpickle is known to have problems
    if pickler_name == 'cloudpickle':
        pytest.skip("'cloudpickle' has issues with 'classproperty'")

    # Import alternative pickler
    _pickler = importlib.import_module(pickler_name)

    def _copy(cls):
        return type(cls.__name__, (cls,), {})

    # Define new classes with different requirements
    @attributes('z')
    @requires('a')
    @requires('b')
    class A:
        pass

    @requires('c,d,e')
    class B(A):

        @property
        def a(self):
            return 1

    @staticvars(b=1)
    class C(B, ClassProperty):
        pass

    @attributes('c', d=2)
    class D(C):
        pass

    @pickler(pickler_name)
    class E(D, Pickler):

        @classproperty
        def e(cls):
            return 42

    # Pickle and unpickle types
    _A = _pickler.loads(_pickler.dumps(A))
    _B = _pickler.loads(_pickler.dumps(B))
    _C = _pickler.loads(_pickler.dumps(C))
    _D = _pickler.loads(_pickler.dumps(D))
    _E = _pickler.loads(_pickler.dumps(E))

    # 'A' requires both a and b
    try:
        A(z=1)
    except AttributeError as e:
        assert ("type object 'A' requires" in str(e))

    # 'A' requires 'b'
    try:
        provides('a')(_copy(A))(z=1)
    except AttributeError as e:
        assert ("type object 'A' requires 'b'" == str(e))

    # 'A' requires 'a'
    try:
        provides('b')(_copy(A))(z=1)
    except AttributeError as e:
        assert ("type object 'A' requires 'a'" == str(e))

    try:
        attributes('a')(_copy(A))(a=1, z=1)
    except AttributeError as e:
        assert ("type object 'A' requires 'b'" == str(e))

    try:
        attributes(b=2)(_copy(A))(z=1)
    except AttributeError as e:
        assert ("type object 'A' requires 'a'" == str(e))

    # Ok
    assert (provides('a')(provides('b')(_copy(A)))(z=-1).z == -1)

    # Ok
    assert (provides('a,b')(_copy(A))(z=-1).z == -1)

    # Ok
    assert (attributes(a=1, b=2)(_copy(A))(z=1).z == 1)

    # Ok
    assert (attributes('a,b')(_copy(A))(a=1, b=2, z=3).a == 1)
    assert (attributes('a,b')(_copy(A))(a=1, b=2, z=3).b == 2)
    assert (attributes('a,b')(_copy(A))(a=1, b=2, z=3).z == 3)

    # Ok
    assert (provides('b,c,d,e')(_copy(B))(z=1).z == 1)

    # Ok
    assert (provides('c,d,e')(_copy(C))(z=1).z == 1)

    # Ok
    assert (provides('e')(_copy(D))(c='a', z=1).c == 'a')

    # Ok
    assert (E(c=1, z=10).z == 10)

    # Class 'E' requires 'c'
    try:
        E(z=1)
    except TypeError as e:
        assert (str(e) == "test__requires.<locals>.E.__init__() missing "
                "required keyword-only argument: 'c'")

    # Class '_E' requires 'c'
    try:
        _E(z=1)
    except TypeError as e:
        assert (str(e) == "test__requires.<locals>.E.__init__() missing "
                "required keyword-only argument: 'c'")

    # Check values
    assert (E.e == 42 and E.e == _E.e)

    # Get objects
    _e1 = E(c='hello!', z=1)
    _e2 = pickle.loads(pickle.dumps(_e1))
    _e3 = _E(c='hi!', z=1)

    # Check values
    assert (_e1.a == 1 and _e1.a == _e2.a == _e3.a)
    assert (_e1.b == 1 and _e1.b == _e2.b == _e3.b)
    assert (_e1.c == 'hello!' and _e3.c == 'hi!' and _e1.c == _e2.c != _e3.c)
    assert (_e1.d == 2 and _e1.d == _e2.d == _e3.d)
    assert (_e1.e == 42 and _e1.e == _e2.e == _e3.e)
    assert (type(_e1).e == 42 and type(_e1).e == type(_e2).e == type(_e3).e)


@pytest.mark.parametrize('pickler_name', ['dill', 'cloudpickle'])
def test__pickler(pickler_name):
    import pickle

    # Cloudpickle is known to have problems
    if pickler_name == 'cloudpickle':
        pytest.skip("'cloudpickle' has issues with 'classproperty'")

    # Let's define a new class
    @attributes('a,b', c=1)
    @staticvars('d', e=-1, transform=dict(d=int, e=str))
    @requires('x')
    class A(ClassProperty, Pickler):

        def __init__(self, y):
            self.y = y

    # Let's define a stateless type
    _A = type(A)('_A', (A,), {})

    # Create a new class, which is inheriting from '_A'
    @pickler(pickler_name)
    class B(_A):

        @property
        def x(self):
            return self.y**2

    # Get a new object
    b = B(a=1, b=2, y=4)

    # Create new
    _b = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(b))))

    # Checks
    assert (b.__dict__ == _b.__dict__)
    assert (b.a == 1 and b.a == _b.a)
    assert (b.b == 2 and b.b == _b.b)
    assert (b.c == 1 and b.c == _b.c)
    try:
        b.d
    except AttributeError as e:
        assert (str(e) == "type object 'B' has no attribute '__A_static_d'")
    try:
        _b.d
    except AttributeError as e:
        assert (str(e) == "type object 'B' has no attribute '__A_static_d'")
    assert (b.e == '-1' and b.e == _b.e)
    assert (b.x == b.y**2 and b.x == _b.x)
    assert (b.y == 4 and b.y == _b.y)


@pytest.mark.parametrize('pickler_name', ['dill', 'cloudpickle'])
def test__hash(pickler_name):
    import pickle

    # 'list' types are not hashable
    try:
        hash(list([1, 2, 3]))
    except TypeError as e:
        assert (str(e) == "unhashable type: 'list'")

    # Howerver, decorama.hash allows makes every type hashable.
    # In this case, the hash value is defined as the sum of all
    # the hashes values of 'A'
    @hasher(method=lambda a: sum(map(hash, a)))
    class A(list):
        pass

    # Check
    assert (hash(A([1, '2', 3.33])) == hash(1) + hash('2') + hash(3.33))

    # By default, decorama.hash uses 'lambda x: hash(pickle.dumps(x))'
    # to get the hash of an arbitrary object
    @hasher
    class A(list):
        pass

    # However, 'A' is a local object and cannot be pickled
    try:
        hash(A([1, 2, 3]))
    except AttributeError as e:
        assert (str(e) == f"Can't pickle local object '{A.__qualname__}'")

    # This problem can be overcome by using decorama.pickled
    @pickler(pickler_name)
    @hasher
    class A(list, Pickler):
        pass

    # Check
    assert (hash(A([1, 2, 3])) == hash(pickle.dumps(A([1, 2, 3]))))

    # If keys are provided, compute hash only using those attributes
    @attributes(a=1, b=2)
    @hasher(keys='a')
    class A:
        ...

    assert (hash(A()) == hash(A()))
    assert (hash(A(b=-3)) == hash(A(b=4)))
    assert (hash(A(a=-3)) != hash(A(a=4)))
    assert (hash(A(a=-3, b='a')) == hash(A(a=-3, b=1)))


@pytest.mark.parametrize('pickler_name', ['dill', 'cloudpickle'])
def test__comparison(pickler_name):
    import pickle

    # Define new class
    @pickler(pickler_name)
    @compare('a', b=lambda x, y: abs(x) == abs(y))
    class A(Pickler):

        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

    # Get new object
    a = A(1, 2, 3)

    # Since 'c' is not appearing in compare, it's ignored
    assert (a == A(1, 2, 'hello!'))

    # 'a' is compared using lambda x,y: x == y (default)
    assert (a != A(-1, 2, 3))

    # 'b' is compared using lambda x,y: abs(x) == abs(y)
    assert (a == A(1, 2, 3))
    assert (a == A(1, -2, 3))
    assert (a != A(1, 3, 3))

    # Compare after pickling
    assert (pickle.loads(pickle.dumps(a)) == a)


@pytest.mark.parametrize('pickler_name', ['dill', 'cloudpickle'])
def test__attributes(pickler_name):
    import pickle

    # Define new class
    @pickler(pickler_name)
    @compare('a,b,c,v,w')
    @attributes('a',
                b=42,
                c=1.23,
                transform=dict(a=int, b=str, c=float),
                check=dict(a=lambda a: a > 0, c=lambda c: isinstance(c, float)))
    class A(Pickler):

        def __init__(self, v, w=None):
            self.v = v
            self.w = w

    # Since 'a' doesn't have a default value, calling A() should raise an error.
    try:
        A()
    except TypeError as e:
        assert (str(e) == "test__attributes.<locals>.A.__init__() missing "
                "required keyword-only argument: 'a'")

    # Also, 'v' must be provided
    try:
        A(a=1.123)
    except TypeError as e:
        assert ("__init__() missing 1 required positional argument: 'v'"
                in str(e))

    # 'a' must be larger than zero, but 'int(a)' is called prior to assign 'a'.
    # Hence, the following should fail.
    try:
        A(v=1, a=0.123)
    except ValueError as e:
        assert (str(e) == "Check failed for variable 'a'")

    # Get new object
    a = A(v=1, a=1.123)

    # Check values
    assert (a.a == int(1.123) and isinstance(a.a, int))
    assert (a.b == str(42) and isinstance(a.b, str))
    assert (a.c == float(1.23) and isinstance(a.c, float))
    assert (a.v == 1)
    assert (a.w == None)

    # Get new object with different values
    a = A(v=1, w=2, a=2.2, b='hello!', c=1.43)

    # Check values
    assert (a.a == int(2.2) and isinstance(a.a, int))
    assert (a.b == "hello!" and isinstance(a.b, str))
    assert (a.c == 1.43 and isinstance(a.c, float))
    assert (a.v == 1)
    assert (a.w == 2)

    # All attributes should be read-only
    for k in 'abc':
        try:
            setattr(a, k, 1)
        except AttributeError as e:
            assert (str(e) == "can't set attribute")

    # Since 'hello!' cannot be converted to float, the following should raise
    try:
        A(v=1, a=2, c='hello!')
    except ValueError as e:
        assert (str(e) == "could not convert string to float: 'hello!'")

    # Try to pickle
    _a = pickle.loads(pickle.dumps(a))

    # Check comparison
    assert (_a == a)

    # Check dictionaries
    assert (_a.__dict__ == a.__dict__)


def test__printer():

    @attributes(a=1, b=2, c=3, d=4)
    @printer(a='a={self.a}',
             b=lambda self: self.b**2,
             f='f=1.23',
             c=PrintObject('nothing', order=10000),
             d=PrintObject('{self.d:1.2f}', order=-1, sep=': '),
             name=PrintObject('NAME', pos='name'),
             pre=PrintObject('[PRE] ', pos='pre'),
             post=PrintObject(' [POST]', pos='post'))
    class A(Printer):
        ...

    @printer(name=PrintObject('NewNAME', pos='name'),
             f=PrintObject('[FIRST]', order=-1, sep=' '),
             h=PrintObject('[FirstPost]', order=-1, pos='post', sep=':'))
    class B(A):
        ...

    assert (str(A()) == "[PRE] NAME(4.00: a=1, 4, f=1.23, nothing) [POST]")
    assert (str(B()) ==
            "[PRE] NewNAME([FIRST] 4.00: a=1, 4, nothing)[FirstPost]: [POST]")


def test__PrintObject():
    try:
        PrintObject(any, order='a')
    except ValueError as e:
        assert (str(e) == "invalid literal for int() with base 10: 'a'")

    try:
        PrintObject(any, pos='a')
    except ValueError as e:
        assert (
            str(e) == "'pos' must be either 'pre', 'name', 'bulk' or 'post'")

    assert (PrintObject(any).fn == any)
    assert (PrintObject(any).pos == 'bulk')
    assert (PrintObject(any).sep == ', ')
    assert (PrintObject(any).order == 100)
    assert (PrintObject(any, pos='pre').pos == 'pre')
    assert (PrintObject(any, order=12345).order == 12345)
