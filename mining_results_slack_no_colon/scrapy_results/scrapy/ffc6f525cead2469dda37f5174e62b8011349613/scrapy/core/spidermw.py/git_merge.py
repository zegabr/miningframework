"""
Spider Middleware manager

See documentation in docs/topics/spider-middleware.rst
"""
import collections.abc
from itertools import islice
from typing import Any, Callable, Generator, Iterable, Union

from twisted.internet.defer import Deferred
from twisted.python.failure import Failure

from scrapy import Request, Spider
from scrapy.exceptions import _InvalidOutput
from scrapy.http import Response
from scrapy.middleware import MiddlewareManager
from scrapy.utils.asyncgen import _process_iterable_universal
from scrapy.utils.conf import build_component_list
from scrapy.utils.defer import mustbe_deferred
from scrapy.utils.python import MutableAsyncChain, MutableChain


<<<<<<< /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/left.py
def _isiterable(possible_iterator):
    return hasattr(possible_iterator, '__iter__') or hasattr(possible_iterator, '__aiter__')
||||||| /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/base.py
def _isiterable(possible_iterator):
    return hasattr(possible_iterator, '__iter__')


def _fname(f):
    return f"{f.__self__.__class__.__name__}.{f.__func__.__name__}"
=======
ScrapeFunc = Callable[[Union[Response, Failure], Request, Spider], Any]


def _isiterable(o) -> bool:
    return isinstance(o, Iterable)
>>>>>>> /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/right.py


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

    def _process_spider_input(self, scrape_func: ScrapeFunc, response: Response, request: Request,
                              spider: Spider) -> Any:
        for method in self.methods['process_spider_input']:
            try:
                result = method(response=response, spider=spider)
                if result is not None:
                    msg = (f"Middleware {method.__qualname__} must return None "
                           f"or raise an exception, got {type(result)}")
                    raise _InvalidOutput(msg)
            except _InvalidOutput:
                raise
            except Exception:
                return scrape_func(Failure(), request, spider)
        return scrape_func(response, request, spider)

<<<<<<< /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/left.py
    def _evaluate_iterable(self, response, spider, iterable, exception_processor_index, recover_to):
        @_process_iterable_universal
        async def _evaluate_async_iterable(iterable):
            try:
                async for r in iterable:
                    yield r
            except Exception as ex:
                exception_result = self._process_spider_exception(response, spider, Failure(ex),
                                                                  exception_processor_index)
                if isinstance(exception_result, Failure):
                    raise
                recover_to.extend(exception_result)
        return _evaluate_async_iterable(iterable)
||||||| /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/base.py
    def _evaluate_iterable(self, response, spider, iterable, exception_processor_index, recover_to):
        try:
            for r in iterable:
                yield r
        except Exception as ex:
            exception_result = self._process_spider_exception(response, spider, Failure(ex),
                                                              exception_processor_index)
            if isinstance(exception_result, Failure):
                raise
            recover_to.extend(exception_result)
=======
    def _evaluate_iterable(self, response: Response, spider: Spider, iterable: Iterable,
                           exception_processor_index: int, recover_to: MutableChain) -> Generator:
        try:
            for r in iterable:
                yield r
        except Exception as ex:
            exception_result = self._process_spider_exception(response, spider, Failure(ex),
                                                              exception_processor_index)
            if isinstance(exception_result, Failure):
                raise
            recover_to.extend(exception_result)
>>>>>>> /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/right.py

    def _process_spider_exception(self, response: Response, spider: Spider, _failure: Failure,
                                  start_index: int = 0) -> Union[Failure, MutableChain]:
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
                msg = (f"Middleware {method.__qualname__} must return None "
                       f"or an iterable, got {type(result)}")
                raise _InvalidOutput(msg)
        return _failure

    def _process_spider_output(self, response: Response, spider: Spider,
                               result: Iterable, start_index: int = 0) -> MutableChain:
        # items in this iterable do not need to go through the process_spider_output
        # chain, they went through it already from the process_spider_exception method
        last_result_async = isinstance(result, collections.abc.AsyncIterator)
        if last_result_async:
            recovered = MutableAsyncChain()
        else:
            recovered = MutableChain()

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
                msg = (f"Middleware {method.__qualname__} must return an "
                       f"iterable, got {type(result)}")
                raise _InvalidOutput(msg)
            if last_result_async and isinstance(result, collections.abc.Iterator):
                raise TypeError(f"Synchronous {method.__qualname__} called with an async iterable")
            last_result_async = isinstance(result, collections.abc.AsyncIterator)

        if last_result_async:
            return MutableAsyncChain(result, recovered)
        else:
            return MutableChain(result, recovered)

<<<<<<< /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/left.py
    def _process_callback_output(self, response, spider, result):
        if isinstance(result, collections.abc.AsyncIterator):
            recovered = MutableAsyncChain()
        else:
            recovered = MutableChain()
||||||| /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/base.py
    def _process_callback_output(self, response, spider, result):
        recovered = MutableChain()
=======
    def _process_callback_output(self, response: Response, spider: Spider, result: Iterable) -> MutableChain:
        recovered = MutableChain()
>>>>>>> /home/ze/miningframework/mining_results_version3_2_no_colon/scrapy_results/scrapy/ffc6f525cead2469dda37f5174e62b8011349613/scrapy/core/spidermw.py/right.py
        result = self._evaluate_iterable(response, spider, result, 0, recovered)
        result = self._process_spider_output(response, spider, result)
        if isinstance(result, collections.abc.AsyncIterator):
            return MutableAsyncChain(result, recovered)
        else:
            return MutableChain(result, recovered)

    def scrape_response(self, scrape_func: ScrapeFunc, response: Response, request: Request,
                        spider: Spider) -> Deferred:
        def process_callback_output(result: Iterable) -> MutableChain:
            return self._process_callback_output(response, spider, result)

        def process_spider_exception(_failure: Failure) -> Union[Failure, MutableChain]:
            return self._process_spider_exception(response, spider, _failure)

        dfd = mustbe_deferred(self._process_spider_input, scrape_func, response, request, spider)
        dfd.addCallbacks(callback=process_callback_output, errback=process_spider_exception)
        return dfd

    def process_start_requests(self, start_requests, spider: Spider) -> Deferred:
        return self._process_chain('process_start_requests', start_requests, spider)