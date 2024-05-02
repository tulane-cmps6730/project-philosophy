from flask import Flask, render_template, request
from plato_matcher_online import process_input  # Make sure this is correctly imported

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    user_input = request.form['user_input']
    summary, citation = process_input(user_input)
    return render_template('results.html', summary=summary, citation=citation)

if __name__ == '__main__':
    app.run(debug=True)