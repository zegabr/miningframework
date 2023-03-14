from markupsafe import escape
from markupsafe import Markup
from werkzeug.exceptions import abort as abort
from werkzeug.utils import redirect as redirect

from . import json as json
from .app import Flask as Flask
from .app import Request as Request
from .app import Response as Response
from .blueprints import Blueprint as Blueprint
from .config import Config as Config
from .ctx import after_this_request as after_this_request
from .ctx import copy_current_request_context as copy_current_request_context
from .ctx import has_app_context as has_app_context
from .ctx import has_request_context as has_request_context
from .globals import _app_ctx_stack as _app_ctx_stack
from .globals import _request_ctx_stack as _request_ctx_stack
from .globals import current_app as current_app
from .globals import g as g
from .globals import request as request
from .globals import session as session
from .helpers import flash as flash
from .helpers import get_flashed_messages as get_flashed_messages
from .helpers import get_template_attribute as get_template_attribute
from .helpers import make_response as make_response
from .helpers import safe_join as safe_join
from .helpers import send_file as send_file
from .helpers import send_from_directory as send_from_directory
from .helpers import stream_with_context as stream_with_context
from .helpers import url_for as url_for
from .json import jsonify as jsonify
from .signals import appcontext_popped as appcontext_popped
from .signals import appcontext_pushed as appcontext_pushed
from .signals import appcontext_tearing_down as appcontext_tearing_down
from .signals import before_render_template as before_render_template
from .signals import got_request_exception as got_request_exception
from .signals import message_flashed as message_flashed
from .signals import request_finished as request_finished
from .signals import request_started as request_started
from .signals import request_tearing_down as request_tearing_down
from .signals import signals_available as signals_available
from .signals import template_rendered as template_rendered
from .templating import render_template as render_template
from .templating import render_template_string as render_template_string

<<<<<<< /home/ze/miningframework/mining_results_version3_3/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/src/flask/__init__.py/left.py
__version__ = "2.0.1.dev0"
||||||| /home/ze/miningframework/mining_results_version3_3/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/src/flask/__init__.py/base.py
    :copyright: 2010 Pallets
    :license: BSD-3-Clause
"""
# utilities we import from Werkzeug and Jinja2 that are unused
# in the module but are exported as public interface.
from jinja2 import escape
from jinja2 import Markup
from werkzeug.exceptions import abort
from werkzeug.utils import redirect

from . import json
from ._compat import json_available
from .app import Flask
from .app import Request
from .app import Response
from .blueprints import Blueprint
from .config import Config
from .ctx import after_this_request
from .ctx import copy_current_request_context
from .ctx import has_app_context
from .ctx import has_request_context
from .globals import _app_ctx_stack
from .globals import _request_ctx_stack
from .globals import current_app
from .globals import g
from .globals import request
from .globals import session
from .helpers import flash
from .helpers import get_flashed_messages
from .helpers import get_template_attribute
from .helpers import make_response
from .helpers import safe_join
from .helpers import send_file
from .helpers import send_from_directory
from .helpers import stream_with_context
from .helpers import url_for
from .json import jsonify
from .signals import appcontext_popped
from .signals import appcontext_pushed
from .signals import appcontext_tearing_down
from .signals import before_render_template
from .signals import got_request_exception
from .signals import message_flashed
from .signals import request_finished
from .signals import request_started
from .signals import request_tearing_down
from .signals import signals_available
from .signals import template_rendered
from .templating import render_template
from .templating import render_template_string

__version__ = "1.1.2"
=======
    :copyright: 2010 Pallets
    :license: BSD-3-Clause
"""
# utilities we import from Werkzeug and Jinja2 that are unused
# in the module but are exported as public interface.
from jinja2 import escape
from jinja2 import Markup
from werkzeug.exceptions import abort
from werkzeug.utils import redirect

from . import json
from ._compat import json_available
from .app import Flask
from .app import Request
from .app import Response
from .blueprints import Blueprint
from .config import Config
from .ctx import after_this_request
from .ctx import copy_current_request_context
from .ctx import has_app_context
from .ctx import has_request_context
from .globals import _app_ctx_stack
from .globals import _request_ctx_stack
from .globals import current_app
from .globals import g
from .globals import request
from .globals import session
from .helpers import flash
from .helpers import get_flashed_messages
from .helpers import get_template_attribute
from .helpers import make_response
from .helpers import safe_join
from .helpers import send_file
from .helpers import send_from_directory
from .helpers import stream_with_context
from .helpers import url_for
from .json import jsonify
from .signals import appcontext_popped
from .signals import appcontext_pushed
from .signals import appcontext_tearing_down
from .signals import before_render_template
from .signals import got_request_exception
from .signals import message_flashed
from .signals import request_finished
from .signals import request_started
from .signals import request_tearing_down
from .signals import signals_available
from .signals import template_rendered
from .templating import render_template
from .templating import render_template_string

__version__ = "1.1.3"
>>>>>>> /home/ze/miningframework/mining_results_version3_3/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/src/flask/__init__.py/right.py
