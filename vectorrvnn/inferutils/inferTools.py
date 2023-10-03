import threading 
import torch

class AutoUnloadModel:
    """ 
    This is a helper class that makes it easy to automatically unload a model from the GPU after a given idle time. 

    The use-case is that you are poor and have just one graphic card that you want to train as well as serve 
    your model on. Now, when you are serving a version of your model, a lot of the time it may be idle, while 
    still occupying VRAM, preventing you to train more models.

    When you inherit from this class, you have to specify 3 methods:

    1. `load`: You control how you load the model. You never have to call this method. We'll make sure that 
    before `infer` is run, the model has been loaded.
    2. `infer`: You use the loaded model to do inference and return the result
    3. `unload`: This is a dummy method that you don't have to worry about. Simply do: 
        
        def unload (self) :
            pass

    Finally, while creating an instance of your derived class of your derived class, you can specify the timeout
    after which you'd like to unload the model. Don't worry, it'll be unloaded the next time `infer` is called.
    """

    def __init__(self, timeout=60):
        self.timer = None
        self.timeout = timeout
        self.lock = threading.Lock() # to make sure we don't unload while loading/inferring
        self.loaded = False

    @classmethod 
    def _unload (cls, unload_fn) : 
        """ Decorator to move all member variables from GPU to CPU and then call empty_cache """
        @wraps(unload_fn) 
        def wrapper(self, *args, **kwargs) : 
            with self.lock : 
                if self.loaded : 
                    print('Unloading model')
                    # FIXME:  In the future, we can check all nn.Module/torch.Tensor types
                    # that are referenced by this object, but for now, this is good enough.
                    for item in dir(self) : 
                        v = getattr(self, item)
                        if 'cpu' in dir(v) and callable(v.cpu) : 
                            v = v.cpu()
                            v = None
                    torch.cuda.empty_cache() # makes sure that the occupied memory is shown as freed in nvidia-smi
                    unload_fn(self, *args, **kwargs)
                    self.loaded = False
        return wrapper

    @classmethod
    def _check_load_and_infer (cls, infer_fn) : 
        """ Decorator that ensures that whatever needs to be loaded is loaded before infer is called """ 
        @wraps(infer_fn) 
        def wrapper (self, *args, **kwargs) : 
            with self.lock : 
                self.reset_timer()
                if not self.loaded : 
                    print('Loading Model')
                    self.load()
                    self.reset_timer()
                    self.loaded = True
                print('Running inference on Model')
                out = infer_fn(self, *args, **kwargs)
                self.reset_timer()
            return out
        return wrapper

    def reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.unload)
        self.timer.start()

    def load(self):
        """ Derived class should implement this but doesn't need to call this because we'll always do this for them """ 
        raise NotImplementedError("Subclasses must implement this method")

    def unload(self):
        """ 
        Derived class should just do something like, so that we can attach our decorator on it.

            def unload (self) :
                pass
        """
        raise NotImplementedError("Subclasses must implement this method")

    def infer(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __init_subclass__(cls, **kwargs):
        """ 
        This is a super cool construct in python. It allows us to modify the behavior of the subclass 
        when it is created. 

        In my use case, I don't really want the subclass to care about loading, unloading, timers, locks
        etc. So I add decorators to the subclass implementations so that all these details are taken care of.
        """
        super().__init_subclass__(**kwargs)
        assert 'infer' in cls.__dict__, "Method infer is not dervied in base class" 
        assert 'load' in cls.__dict__, "Method load is not dervied in base class" 
        assert 'unload' in cls.__dict__, "Method unload is not dervied in base class" 

        setattr(cls, 'infer', cls._check_load_and_infer(cls.__dict__['infer']))
        setattr(cls, 'unload', cls._unload(cls.__dict__['unload']))

