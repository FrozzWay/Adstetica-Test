from flask import (
    request,
    Blueprint,
    jsonify
)

from src.neuronet import nnet

bp = Blueprint('api', __name__)


@bp.get("/analise")
def analise():
    text = request.args.get('text')
    lang, topic = nnet.analise(text)
    response = {
        "lang": lang,
        "topic": topic
    }
    return jsonify(response)
