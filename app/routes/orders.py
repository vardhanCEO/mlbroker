from flask import Blueprint, render_template
from flask_login import login_required

orders_bp = Blueprint('orders', __name__)


@orders_bp.route('/')
@login_required
def index():
    return render_template('orders/index.html')
