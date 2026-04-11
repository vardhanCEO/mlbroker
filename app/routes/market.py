from flask import Blueprint, render_template
from flask_login import login_required

market_bp = Blueprint('market', __name__)


@market_bp.route('/')
@login_required
def index():
    return render_template('market/index.html')
