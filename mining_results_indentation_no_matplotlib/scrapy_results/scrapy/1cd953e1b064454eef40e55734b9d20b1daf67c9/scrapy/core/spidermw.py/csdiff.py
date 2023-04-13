"""
Spider Middleware manager

See documentation in docs/topics/spider-middleware.rst
"""
import inspect
from itertools import islice

from twisted.python.failure import Failure

from scrapy.exceptions import _InvalidOutput
from scrapy.middleware import MiddlewareManager
from scrapy.utils.conf import build_component_list
from scrapy.utils.defer import mustbe_deferred
from scrapy.utils.python import MutableAsyncChain, MutableChain


def _isiterable(possible_iterator):
    return hasattr(possible_iterator, '__iter__') or hasattr(possible_iterator, '__aiter__')


def _fname(f):
    return f"{f.__self__.__class__.__name__}.{f.__func__.__name__}"


class SpiderMiddlewareManager(MiddlewareManager):

    component_name = 'spider middleware'

    @classmethod
    def _get_mwlist_from_settings(cls, settings):
        return build_component_list(settings.getwithbase('SPIDER_MIDDLEWARES'))

    def _add_middleware(self, mw):
        super()._add_middleware(mw)
        if hasattr(mw, 'process_spider_input'):
            self.methods['process_spider_input'].append(mw.process_spider_input)
        if hasattr(mw, 'process_start_requests'):
            self.methods['process_start_requests'].appendleft(mw.process_start_requests)
        process_spider_output = getattr(mw, 'process_spider_output', None)
        self.methods['process_spider_output'].appendleft(process_spider_output)
        process_spider_exception = getattr(mw, 'process_spider_exception', None)
        self.methods['process_spider_exception'].appendleft(process_spider_exception)

    def _process_spider_input(self, scrape_func, response, request, spider):
        for method in self.methods['process_spider_input']:
            try:
                result = method(response=response, spider=spider)
                if result is not None:
                    msg = (f"Middleware {_fname(method)} must return None "
                           f"or raise an exception, got {type(result)}")
                    raise _InvalidOutput(msg)
            except _InvalidOutput:
                raise
            except Exception:
                return scrape_func(Failure(), request, spider)
        return scrape_func(response, request, spider)

    def _evaluate_iterable(self, response, spider, iterable, exception_processor_index, recover_to):
<<<<<<< /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/left.py
            def _process_exception
=======
        try
>>>>>>> /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/right.py
<<<<<<< /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/left.py
(ex)
=======
:
            for r in iterable:
                yield r
        except Exception as ex
>>>>>>> /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/right.py
:
            exception_result = self._process_spider_exception(response, spider, Failure(ex),
                                                              exception_processor_index)
            if isinstance(exception_result, Failure):
                raise
            recover_to.extend(exception_result)

<<<<<<< /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/left.py
            def _evaluate_normal_iterable(iterable):
                try:
                    for r in iterable:
                        yield r
                except Exception as ex:
                    _process_exception(ex)

            async def _evaluate_async_iterable(iterable):
                try:
                    async for r in iterable:
                        yield r
                except Exception as ex:
                    _process_exception(ex)

            if inspect.isasyncgen(iterable):
                return _evaluate_async_iterable(iterable)
            else:
                return _evaluate_normal_iterable(iterable)

        def process_spider_exception
=======
    def _process_spider_exception
>>>>>>> /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/right.py
(self, response, spider, _failure, start_index=0):
        exception = _failure.value
        # don't handle _InvalidOutput exception
        if isinstance(exception, _InvalidOutput):
            return _failure
        method_list = islice(self.methods['process_spider_exception'], start_index, None)
        for method_index, method in enumerate(method_list, start=start_index):
            if method is None:
                continue
            result = method(response=response, exception=exception, spider=spider)
            if _isiterable(result):
                # stop exception handling by handing control over to the
                # process_spider_output chain if an iterable has been returned
                return self._process_spider_output(response, spider, result, method_index + 1)
            elif result is None:
                continue
            else:
                msg = (f"Middleware {_fname(method)} must return None "
                       f"or an iterable, got {type(result)}")
                raise _InvalidOutput(msg)
        return _failure

    def _process_spider_output(self, response, spider, result, start_index=0):
        # items in this iterable do not need to go through the process_spider_output
        # chain, they went through it already from the process_spider_exception method
<<<<<<< /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/left.py
            if inspect.isasyncgen(result):
                iter_class = MutableAsyncChain
            else:
                iter_class = MutableChain
            recovered = iter_class
=======
        recovered = MutableChain
>>>>>>> /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/right.py
()

        method_list = islice(self.methods['process_spider_output'], start_index, None)
        for method_index, method in enumerate(method_list, start=start_index):
            if method is None:
                continue
            try:
                # might fail directly if the output value is not a generator
                result = method(response=response, result=result, spider=spider)
            except Exception as ex:
                exception_result = self._process_spider_exception(response, spider, Failure(ex), method_index + 1)
                if isinstance(exception_result, Failure):
                    raise
                return exception_result
            if _isiterable(result):
                result = self._evaluate_iterable(response, spider, result, method_index + 1, recovered)
            else:
                msg = (f"Middleware {_fname(method)} must return an "
                       f"iterable, got {type(result)}")
                raise _InvalidOutput(msg)

<<<<<<< /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/left.py
            return iter_class
=======
        return MutableChain
>>>>>>> /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/right.py
(result, recovered)

    def _process_callback_output(self, response, spider, result):
<<<<<<< /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/left.py
            if inspect.isasyncgen(result):
                iter_class = MutableAsyncChain
            else:
                iter_class = MutableChain
            recovered = iter_class
=======
        recovered = MutableChain
>>>>>>> /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/right.py
()
        result = self._evaluate_iterable(response, spider, result, 0, recovered)
<<<<<<< /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/left.py
            return iter_class
=======
        return MutableChain(self._process_spider_output(response, spider, result), recovered)

    def scrape_response
>>>>>>> /home/ze/miningframework/mining_results_indentation/scrapy_results/scrapy/1cd953e1b064454eef40e55734b9d20b1daf67c9/scrapy/core/spidermw.py/right.py
(self, scrape_func, response, request, spider):
        def process_callback_output(result):
            return self._process_callback_output(response, spider, result)

        def process_spider_exception(_failure):
            return self._process_spider_exception(response, spider, _failure)

        dfd = mustbe_deferred(self._process_spider_input, scrape_func, response, request, spider)
        dfd.addCallbacks(callback=process_callback_output, errback=process_spider_exception)
        return dfd

    def process_start_requests(self, start_requests, spider):
        return self._process_chain('process_start_requests', start_requests, spider)
