class Foo:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def detail(self):
        print(self.name)
        print(self.age)

obj1 = Foo('changed', 18)
obj2 = Foo('python', 99)

obj1.detail()
obj2.detail()
