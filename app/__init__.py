from flask import Flask
from config import Config
from .extensions import db, migrate, login_manager


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    with app.app_context():
        from .models.user import User                    # noqa
        from .models.strategy import Strategy, Signal  # noqa
        from .models.order import Order                # noqa
        from .models.binary_trade import BinaryTrade   # noqa
        from .models.bot_trade import BotTrade         # noqa
        from .models.ml_model import MLModel, MLSignal # noqa
        db.create_all()

    from .routes.auth import auth_bp
    from .routes.dashboard import dashboard_bp
    from .routes.strategies import strategies_bp
    from .routes.market import market_bp
    from .routes.orders import orders_bp
    from .routes.api import api_bp

    app.register_blueprint(auth_bp,        url_prefix='/auth')
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(strategies_bp,  url_prefix='/strategies')
    app.register_blueprint(market_bp,      url_prefix='/market')
    app.register_blueprint(orders_bp,      url_prefix='/orders')
    app.register_blueprint(api_bp,         url_prefix='/api')

    from .routes.bot import bot_bp
    app.register_blueprint(bot_bp, url_prefix='/bot')

    from .services.bot_engine import recover_pending
    recover_pending(app)

    return app
