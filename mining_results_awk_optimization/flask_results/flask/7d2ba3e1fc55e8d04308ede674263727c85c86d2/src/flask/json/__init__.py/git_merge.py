import io
import json as _json
import uuid
import warnings
from datetime import date
from datetime import datetime

from markupsafe import Markup
from werkzeug.http import http_date

from ..globals import current_app
from ..globals import request

try:
    import dataclasses
except ImportError:
    # Python < 3.7
    dataclasses = None


class JSONEncoder(_json.JSONEncoder):
    """The default JSON encoder. Handles extra types compared to the
    built-in :class:`json.JSONEncoder`.

    -   :class:`datetime.datetime` and :class:`datetime.date` are
        serialized to :rfc:`822` strings. This is the same as the HTTP
        date format.
    -   :class:`uuid.UUID` is serialized to a string.
    -   :class:`dataclasses.dataclass` is passed to
        :func:`dataclasses.asdict`.
    -   :class:`~markupsafe.Markup` (or any object with a ``__html__``
        method) will call the ``__html__`` method to get a string.

    Assign a subclass of this to :attr:`flask.Flask.json_encoder` or
    :attr:`flask.Blueprint.json_encoder` to override the default.
    """

    def default(self, o):
        """Convert ``o`` to a JSON serializable type. See
        :meth:`json.JSONEncoder.default`. Python does not support
        overriding how basic types like ``str`` or ``list`` are
        serialized, they are handled before this method.
        """
        if isinstance(o, datetime):
            return http_date(o.utctimetuple())
        if isinstance(o, date):
            return http_date(o.timetuple())
        if isinstance(o, uuid.UUID):
            return str(o)
        if dataclasses and dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if hasattr(o, "__html__"):
            return str(o.__html__())
        return super().default(self, o)


class JSONDecoder(_json.JSONDecoder):
    """The default JSON decoder.

    This does not change any behavior from the built-in
    :class:`json.JSONDecoder`.

    Assign a subclass of this to :attr:`flask.Flask.json_decoder` or
    :attr:`flask.Blueprint.json_decoder` to override the default.
    """


def _dump_arg_defaults(kwargs, app=None):
    """Inject default arguments for dump functions."""
    if app is None:
        app = current_app

    if app:
        bp = app.blueprints.get(request.blueprint) if request else None
        cls = bp.json_encoder if bp and bp.json_encoder else app.json_encoder
        kwargs.setdefault("cls", cls)
        kwargs.setdefault("ensure_ascii", app.config["JSON_AS_ASCII"])
        kwargs.setdefault("sort_keys", app.config["JSON_SORT_KEYS"])
    else:
        kwargs.setdefault("sort_keys", True)
        kwargs.setdefault("cls", JSONEncoder)


def _load_arg_defaults(kwargs, app=None):
    """Inject default arguments for load functions."""
    if app is None:
        app = current_app

    if app:
        bp = app.blueprints.get(request.blueprint) if request else None
        cls = bp.json_decoder if bp and bp.json_decoder else app.json_decoder
        kwargs.setdefault("cls", cls)
    else:
        kwargs.setdefault("cls", JSONDecoder)


def dumps(obj, app=None, **kwargs):
    """Serialize an object to a string of JSON.

    Takes the same arguments as the built-in :func:`json.dumps`, with
    some defaults from application configuration.

    :param obj: Object to serialize to JSON.
    :param app: Use this app's config instead of the active app context
        or defaults.
    :param kwargs: Extra arguments passed to func:`json.dumps`.

    .. versionchanged:: 2.0
        ``encoding`` is deprecated and will be removed in 2.1.

    .. versionchanged:: 1.0.3
        ``app`` can be passed directly, rather than requiring an app
        context for configuration.
    """
    _dump_arg_defaults(kwargs, app=app)
    encoding = kwargs.pop("encoding", None)
    rv = _json.dumps(obj, **kwargs)

    if encoding is not None:
        warnings.warn(
            "'encoding' is deprecated and will be removed in 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )

        if isinstance(rv, str):
            return rv.encode(encoding)

    return rv


def dump(obj, fp, app=None, **kwargs):
    """Serialize an object to JSON written to a file object.

    Takes the same arguments as the built-in :func:`json.dump`, with
    some defaults from application configuration.

    :param obj: Object to serialize to JSON.
    :param fp: File object to write JSON to.
    :param app: Use this app's config instead of the active app context
        or defaults.
    :param kwargs: Extra arguments passed to func:`json.dump`.

    .. versionchanged:: 2.0
        Writing to a binary file, and the ``encoding`` argument, is
        deprecated and will be removed in 2.1.
    """
    _dump_arg_defaults(kwargs, app=app)
    encoding = kwargs.pop("encoding", None)
    show_warning = encoding is not None

    try:
        fp.write("")
    except TypeError:
        show_warning = True
        fp = io.TextIOWrapper(fp, encoding or "utf-8")

    if show_warning:
        warnings.warn(
            "Writing to a binary file, and the 'encoding' argument, is"
            " deprecated and will be removed in 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )

    _json.dump(obj, fp, **kwargs)


def loads(s, app=None, **kwargs):
    """Deserialize an object from a string of JSON.

    Takes the same arguments as the built-in :func:`json.loads`, with
    some defaults from application configuration.

    :param s: JSON string to deserialize.
<<<<<<< /home/ze/miningframework/mining_results_version3_3/flask_results/flask/7d2ba3e1fc55e8d04308ede674263727c85c86d2/src/flask/json/__init__.py/left.py
    :param app: Use this app's config instead of the active app context
        or defaults.
    :param kwargs: Extra arguments passed to func:`json.dump`.
||||||| /home/ze/miningframework/mining_results_version3_3/flask_results/flask/7d2ba3e1fc55e8d04308ede674263727c85c86d2/src/flask/json/__init__.py/base.py
    :param app: App instance to use to configure the JSON decoder.
        Uses ``current_app`` if not given, and falls back to the default
        encoder when not in an app context.
    :param kwargs: Extra arguments passed to :func:`json.dumps`.
=======
    :param app: App instance to use to configure the JSON decoder.
        Uses ``current_app`` if not given, and falls back to the default
        encoder when not in an app context.
    :param kwargs: Extra arguments passed to :func:`json.loads`.
>>>>>>> /home/ze/miningframework/mining_results_version3_3/flask_results/flask/7d2ba3e1fc55e8d04308ede674263727c85c86d2/src/flask/json/__init__.py/right.py

    .. versionchanged:: 2.0
        ``encoding`` is deprecated and will be removed in 2.1. The data
        must be a string or UTF-8 bytes.

    .. versionchanged:: 1.0.3
        ``app`` can be passed directly, rather than requiring an app
        context for configuration.
    """
    _load_arg_defaults(kwargs, app=app)
    encoding = kwargs.pop("encoding", None)

    if encoding is not None:
        warnings.warn(
            "'encoding' is deprecated and will be removed in 2.1. The"
            " data must be a string or UTF-8 bytes.",
            DeprecationWarning,
            stacklevel=2,
        )

        if isinstance(s, bytes):
            s = s.decode(encoding)

    return _json.loads(s, **kwargs)


def load(fp, app=None, **kwargs):
    """Deserialize an object from JSON read from a file object.

    Takes the same arguments as the built-in :func:`json.load`, with
    some defaults from application configuration.

    :param fp: File object to read JSON from.
    :param app: Use this app's config instead of the active app context
        or defaults.
    :param kwargs: Extra arguments passed to func:`json.load`.

    .. versionchanged:: 2.0
        ``encoding`` is deprecated and will be removed in 2.1. The file
        must be text mode, or binary mode with UTF-8 bytes.
    """
    _load_arg_defaults(kwargs, app=app)
    encoding = kwargs.pop("encoding", None)

    if encoding is not None:
        warnings.warn(
            "'encoding' is deprecated and will be removed in 2.1. The"
            " file must be text mode, or binary mode with UTF-8 bytes.",
            DeprecationWarning,
            stacklevel=2,
        )

        if isinstance(fp.read(0), bytes):
            fp = io.TextIOWrapper(fp, encoding)

    return _json.load(fp, **kwargs)


_htmlsafe_map = str.maketrans(
    {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026", "'": "\\u0027"}
)


def htmlsafe_dumps(obj, **kwargs):
    """Serialize an object to a string of JSON, replacing HTML-unsafe
    characters with Unicode escapes. Otherwise behaves the same as
    :func:`dumps`.

    This is available in templates as the ``|tojson`` filter, which will
    also mark the result with ``|safe``.

    The returned string is safe to render in HTML documents and
    ``<script>`` tags. The exception is in HTML attributes that are
    double quoted; either use single quotes or the ``|forceescape``
    filter.

    .. versionchanged:: 0.10
        Single quotes are escaped, making this safe to use in HTML,
        ``<script>`` tags, and single-quoted attributes without further
        escaping.
    """
    return dumps(obj, **kwargs).translate(_htmlsafe_map)


def htmlsafe_dump(obj, fp, **kwargs):
    """Serialize an object to JSON written to a file object, replacing
    HTML-unsafe characters with Unicode escapes. See
    :func:`htmlsafe_dumps` and :func:`dumps`.
    """
    fp.write(htmlsafe_dumps(obj, **kwargs))


def jsonify(*args, **kwargs):
    """Serialize data to JSON and wrap it in a :class:`~flask.Response`
    with the :mimetype:`application/json` mimetype.

    Uses :func:`dumps` to serialize the data, but ``args`` and
    ``kwargs`` are treated as data rather than arguments to
    :func:`json.dumps`.

    1.  Single argument: Treated as a single value.
    2.  Multiple arguments: Treated as a list of values.
        ``jsonify(1, 2, 3)`` is the same as ``jsonify([1, 2, 3])``.
    3.  Keyword arguments: Treated as a dict of values.
        ``jsonify(data=data, errors=errors)`` is the same as
        ``jsonify({"data": data, "errors": errors})``.
    4.  Passing both arguments and keyword arguments is not allowed as
        it's not clear what should happen.

    .. code-block:: python

        from flask import jsonify

        @app.route("/users/me")
        def get_current_user():
            return jsonify(
                username=g.user.username,
                email=g.user.email,
                id=g.user.id,
            )

    Will return a JSON response like this:

    .. code-block:: javascript

        {
          "username": "admin",
          "email": "admin@localhost",
          "id": 42
        }

    The default output omits indents and spaces after separators. In
    debug mode or if :data:`JSONIFY_PRETTYPRINT_REGULAR` is ``True``,
    the output will be formatted to be easier to read.

    .. versionchanged:: 0.11
        Added support for serializing top-level arrays. This introduces
        a security risk in ancient browsers. See :ref:`security-json`.

    .. versionadded:: 0.2
    """
    indent = None
    separators = (",", ":")

    if current_app.config["JSONIFY_PRETTYPRINT_REGULAR"] or current_app.debug:
        indent = 2
        separators = (", ", ": ")

    if args and kwargs:
        raise TypeError("jsonify() behavior undefined when passed both args and kwargs")
    elif len(args) == 1:  # single args are passed directly to dumps()
        data = args[0]
    else:
        data = args or kwargs

    return current_app.response_class(
        f"{dumps(data, indent=indent, separators=separators)}\n",
        mimetype=current_app.config["JSONIFY_MIMETYPE"],
    )


def tojson_filter(obj, **kwargs):
    return Markup(htmlsafe_dumps(obj, **kwargs))