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

<<<<<<< /home/ze/miningframework/mining_results/flask_results/flask/7161776824734fc2797fe2b4fc974d183487ebf8/src/flask/__init__.py/left.py
__version__ = "2.1.0.dev0"
=======
__version__ = "2.0.2.dev0"
>>>>>>> /home/ze/miningframework/mining_results/flask_results/flask/7161776824734fc2797fe2b4fc974d183487ebf8/src/flask/__init__.py/right.py
