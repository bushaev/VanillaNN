class Parameter():
    def __init__(self, param):
        self.param = param
        self.grad_ = None

    #TODO: Try to work around that
    def get(self):
        return self.param

    def set(self, param):
        self.param = param

    def minus_(self, other):
        self.param = self.param - other

    def plus_(self, other):
        self.param = self.param + other

    def set_grad(self, grad):
        self.grad_ = grad

    def grad(self):
        return self.grad_