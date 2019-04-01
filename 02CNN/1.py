class Foo:
    def bar(self):
        print('Bar')

    def hello(self, name):
        print('I am %s' %name)


obj = Foo()
obj.bar()
obj.hello('changed')
